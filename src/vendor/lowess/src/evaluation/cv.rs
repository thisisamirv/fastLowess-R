//! Cross-validation for LOWESS bandwidth selection.
//!
//! ## Purpose
//!
//! This module provides cross-validation tools for selecting the optimal
//! smoothing fraction (bandwidth) in LOWESS regression. It implements
//! generic k-fold and leave-one-out cross-validation strategies that work
//! with any smoothing function.
//!
//! ## Design notes
//!
//! * Cross-validation is generic over the smoothing implementation via callbacks.
//! * Supports both k-fold CV and leave-one-out CV (LOOCV).
//! * Uses linear interpolation to predict on held-out test data.
//! * Selects the fraction that minimizes root mean squared error (RMSE).
//! * All operations are generic over `Float` types to support f32 and f64.
//! * Supports both `std` and `no_std` environments.
//!
//! ## Available methods
//!
//! * **K-Fold CV**: Partitions data into k subsamples, training on k-1 and validating on 1
//! * **Leave-One-Out CV (LOOCV)**: Extreme case of k-fold where k equals sample size
//!
//! ## Key concepts
//!
//! ### Cross-Validation
//! Cross-validation estimates prediction error by repeatedly fitting the model
//! on training subsets and evaluating on held-out test subsets. The fraction
//! with the lowest average error is selected.
//!
//! ### K-Fold Strategy
//! Data is partitioned into k equal-sized folds. Each fold serves as the test
//! set once while the remaining k-1 folds form the training set. This provides
//! a good balance between computational cost and accuracy.
//!
//! ### Leave-One-Out Strategy
//! Each data point is held out once as the test set while all other points
//! form the training set. This is the most accurate but also the most
//! computationally expensive method (n iterations).
//!
//! ### Interpolation for Prediction
//! Since test points may not be in the training set, predictions are made
//! by linearly interpolating the smoothed training values. Constant
//! extrapolation is used at boundaries.
//!
//! ## Invariants
//!
//! * Training and test sets are disjoint in each fold.
//! * All data points are used for testing exactly once.
//! * The best fraction minimizes RMSE across all folds.
//! * Interpolation is well-defined for sorted x values.
//!
//! ## Non-goals
//!
//! * This module does not perform the actual smoothing (done via callback).
//! * This module does not validate input data or fraction ranges.
//! * This module does not support stratified or time-series CV.
//! * This module does not provide confidence intervals for CV scores.
//!
//! ## Visibility
//!
//! The [`CVMethod`] enum is part of the public API, allowing users to
//! configure cross-validation strategies. The internal implementation
//! details may change without notice.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// Cross-Validation Configuration
// ============================================================================

/// Cross-validation method configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVMethod {
    /// K-fold cross-validation with k folds.
    KFold(usize),

    /// Leave-one-out cross-validation.
    LOOCV,
}

impl CVMethod {
    // ========================================================================
    // Public API
    // ========================================================================

    /// Run cross-validation to select the best fraction.
    pub fn run<T, F>(self, x: &[T], y: &[T], fractions: &[T], smoother: F) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T> + Copy,
    {
        match self {
            CVMethod::KFold(k) => Self::kfold_cross_validation(x, y, fractions, k, smoother),
            CVMethod::LOOCV => Self::leave_one_out_cross_validation(x, y, fractions, smoother),
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Build a data subset from a list of indices.
    pub fn build_subset_from_indices<T: Float>(
        x: &[T],
        y: &[T],
        indices: &[usize],
    ) -> (Vec<T>, Vec<T>) {
        let mut tx = Vec::with_capacity(indices.len());
        let mut ty = Vec::with_capacity(indices.len());
        for &i in indices {
            tx.push(x[i]);
            ty.push(y[i]);
        }
        (tx, ty)
    }

    /// Interpolate prediction at a new x value given fitted training values.
    ///
    /// # Implementation notes
    ///
    /// * Uses binary search for O(log n) bracketing
    /// * Handles duplicate x-values by averaging y-values
    /// * Constant extrapolation prevents unbounded predictions
    pub fn interpolate_prediction<T: Float>(x_train: &[T], y_train: &[T], x_new: T) -> T {
        let n = x_train.len();

        // Edge case: empty training set
        if n == 0 {
            return T::zero();
        }

        // Edge case: single training point
        if n == 1 {
            return y_train[0];
        }

        // Boundary handling: constant extrapolation
        if x_new <= x_train[0] {
            return y_train[0];
        }
        if x_new >= x_train[n - 1] {
            return y_train[n - 1];
        }

        // Binary search for bracketing points
        let mut left = 0;
        let mut right = n - 1;

        while right - left > 1 {
            let mid = (left + right) / 2;
            if x_train[mid] <= x_new {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation between left and right
        let x0 = x_train[left];
        let x1 = x_train[right];
        let y0 = y_train[left];
        let y1 = y_train[right];

        let denom = x1 - x0;
        if denom <= T::zero() {
            // Duplicate x-values: return average of y-values
            return (y0 + y1) / T::from(2.0).unwrap();
        }

        let alpha = (x_new - x0) / denom;
        y0 + alpha * (y1 - y0)
    }

    // ========================================================================
    // Internal Cross-Validation Implementations
    // ========================================================================

    /// Select the best fraction based on cross-validation scores.
    fn select_best_fraction<T: Float>(fractions: &[T], scores: &[T]) -> (T, Vec<T>) {
        if fractions.is_empty() {
            return (T::zero(), Vec::new());
        }

        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (fractions[best_idx], scores.to_vec())
    }

    /// Perform k-fold cross-validation.
    fn kfold_cross_validation<T, F>(
        x: &[T],
        y: &[T],
        fractions: &[T],
        k: usize,
        smoother: F,
    ) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T>,
    {
        if fractions.is_empty() || k < 2 {
            let default = T::from(0.67).unwrap_or(T::from(0.5).unwrap());
            return (default, vec![T::zero()]);
        }

        let n = x.len();
        let fold_size = n / k;

        let mut cv_scores = vec![T::zero(); fractions.len()];

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            // Store RMSE for each fold, then compute mean
            let mut fold_rmses = Vec::with_capacity(k);

            for fold in 0..k {
                // Define test set for this fold
                let test_start = fold * fold_size;
                let test_end = if fold == k - 1 {
                    n // Last fold includes remainder
                } else {
                    (fold + 1) * fold_size
                };

                // Build training set (all indices except test set)
                let train_indices: Vec<usize> = (0..n)
                    .filter(|&i| i < test_start || i >= test_end)
                    .collect();
                let (train_x, train_y) = Self::build_subset_from_indices(x, y, &train_indices);

                // Fit smoother on training data
                let train_smooth = smoother(&train_x, &train_y, frac);

                // Compute RMSE on test set using interpolation
                let mut fold_error = T::zero();
                let mut fold_count = T::zero();
                for i in test_start..test_end {
                    let predicted = Self::interpolate_prediction(&train_x, &train_smooth, x[i]);
                    let error = y[i] - predicted;
                    fold_error = fold_error + error * error;
                    fold_count = fold_count + T::one();
                }

                // Compute RMSE for this fold
                if fold_count > T::zero() {
                    fold_rmses.push((fold_error / fold_count).sqrt());
                }
            }

            // Compute mean of fold RMSEs (matches sklearn's cross_val_score)
            if !fold_rmses.is_empty() {
                let sum: T = fold_rmses.iter().copied().fold(T::zero(), |a, b| a + b);
                cv_scores[frac_idx] = sum / T::from(fold_rmses.len()).unwrap();
            } else {
                cv_scores[frac_idx] = T::infinity();
            }
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }

    /// Perform leave-one-out cross-validation (LOOCV).
    fn leave_one_out_cross_validation<T, F>(
        x: &[T],
        y: &[T],
        fractions: &[T],
        smoother: F,
    ) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T>,
    {
        if fractions.is_empty() {
            let default = T::from(0.67).unwrap_or(T::from(0.5).unwrap());
            return (default, vec![T::zero()]);
        }

        let n = x.len();
        if n < 2 {
            return (fractions[0], vec![T::zero(); fractions.len()]);
        }

        let mut cv_scores = vec![T::zero(); fractions.len()];

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            let mut total_error = T::zero();

            for i in 0..n {
                // Build leave-one-out training set (all points except i)
                let train_indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                let (train_x, train_y) = Self::build_subset_from_indices(x, y, &train_indices);

                // Fit smoother on training data
                let train_smooth = smoother(&train_x, &train_y, frac);

                // Predict at held-out point
                let predicted = Self::interpolate_prediction(&train_x, &train_smooth, x[i]);
                let error = y[i] - predicted;
                total_error = total_error + error * error;
            }

            // Compute RMSE for this fraction
            cv_scores[frac_idx] = (total_error / T::from(n).unwrap()).sqrt();
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }
}
