//! Output types and result structures for LOWESS operations.
//!
//! ## Purpose
//!
//! This module defines the [`LowessResult`] struct which encapsulates all
//! outputs from a LOWESS smoothing operation. It provides a comprehensive
//! container for smoothed values, diagnostics, confidence/prediction intervals,
//! and metadata about the smoothing process.
//!
//! ## Design notes
//!
//! * All optional outputs use `Option<Vec<T>>` for memory efficiency.
//! * Results are generic over `Float` types to support f32 and f64.
//! * Provides convenience methods for common queries.
//! * Implements `Display` for human-readable output with adaptive formatting.
//! * Sorted x-values are stored to maintain correspondence with outputs.
//! * All vectors have the same length (number of data points).
//!
//! ## Available outputs
//!
//! * **Core outputs**: Sorted x-values, smoothed y-values
//! * **Uncertainty**: Standard errors, confidence intervals, prediction intervals
//! * **Diagnostics**: RMSE, MAE, R^2, AIC, AICc, effective DF
//! * **Residuals**: Differences between original and smoothed values
//! * **Robustness**: Final robustness weights from iterative refinement
//! * **Metadata**: Fraction used, iterations performed, CV scores
//!
//! ## Key concepts
//!
//! ### Optional Outputs
//!
//! Most results are optional and only populated when specific features are
//! enabled (e.g., standard errors, diagnostics). This minimizes memory usage
//! and computation time for basic smoothing tasks.
//!
//! ### Intervals
//!
//! * **Confidence Intervals**: Quantify uncertainty in the fitted mean curve:
//!   y_hat +/- z * SE.
//! * **Prediction Intervals**: Quantify uncertainty for individual new observations,
//!   accounting for both estimation error and residual noise:
//!   y_hat +/- z * sqrt(SE^2 + sigma^2).
//!
//! ### Iteration and Selection Metadata
//!
//! When auto-convergence is used, `iterations_used` provides the actual count
//! of robustness passes. When cross-validation is used, `fraction_used` reflects
//! the optimal bandwidth selected, and `cv_scores` provide the RMSE values for
//! all candidate fractions.
//!
//! ## Invariants
//!
//! * All populated vectors have the same length as the input data.
//! * x-values are sorted in monotonically increasing order.
//! * Lower bounds are always less than or equal to upper bounds for all intervals.
//! * Robustness weights are always in the range [0, 1].
//!
//! ## Non-goals
//!
//! * This module does not perform calculations; it only stores results.
//! * This module does not validate result consistency (responsibility of the engine).
//! * This module does not provide serialization/deserialization logic.
//!
//! ## Visibility
//!
//! The [`LowessResult`] struct is part of the public API and is the primary
//! result type returned by all LOWESS adapters.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use core::cmp::Ordering;
use num_traits::Float;

use crate::evaluation::diagnostics::Diagnostics;

// ============================================================================
// Result Structure
// ============================================================================

/// Comprehensive LOWESS result containing smoothed values and diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct LowessResult<T> {
    /// Sorted x-values (independent variable).
    pub x: Vec<T>,

    /// Smoothed y-values (dependent variable).
    pub y: Vec<T>,

    /// Standard errors of the fit at each point.
    pub standard_errors: Option<Vec<T>>,

    /// Lower bounds of the confidence intervals for the mean response.
    pub confidence_lower: Option<Vec<T>>,

    /// Upper bounds of the confidence intervals for the mean response.
    pub confidence_upper: Option<Vec<T>>,

    /// Lower bounds of the prediction intervals for new observations.
    pub prediction_lower: Option<Vec<T>>,

    /// Upper bounds of the prediction intervals for new observations.
    pub prediction_upper: Option<Vec<T>>,

    /// Residuals from the fit (y_i - y_hat_i).
    pub residuals: Option<Vec<T>>,

    /// Final robustness weights from the iterative refinement process.
    pub robustness_weights: Option<Vec<T>>,

    /// Comprehensive diagnostic metrics (RMSE, R^2, AIC, etc.).
    pub diagnostics: Option<Diagnostics<T>>,

    /// Number of robustness iterations actually performed.
    pub iterations_used: Option<usize>,

    /// Smoothing fraction used for the fit (optimal if selected by CV).
    pub fraction_used: T,

    /// RMSE scores for each tested fraction during cross-validation.
    pub cv_scores: Option<Vec<T>>,
}

impl<T: Float> LowessResult<T> {
    // ========================================================================
    // Query Methods
    // ========================================================================

    /// Check if confidence intervals were computed.
    pub fn has_confidence_intervals(&self) -> bool {
        self.confidence_lower.is_some() && self.confidence_upper.is_some()
    }

    /// Check if prediction intervals were computed.
    pub fn has_prediction_intervals(&self) -> bool {
        self.prediction_lower.is_some() && self.prediction_upper.is_some()
    }

    /// Check if cross-validation was performed.
    pub fn has_cv_scores(&self) -> bool {
        self.cv_scores.is_some()
    }

    /// Get the best (minimum) CV score.
    pub fn best_cv_score(&self) -> Option<T> {
        self.cv_scores.as_ref().and_then(|scores| {
            scores
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        })
    }
}

// ============================================================================
// Display Implementation
// ============================================================================

impl<T: Float + core::fmt::Display> core::fmt::Display for LowessResult<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Summary:")?;
        writeln!(f, "  Data points: {}", self.x.len())?;
        writeln!(f, "  Fraction: {}", self.fraction_used)?;

        if let Some(iters) = self.iterations_used {
            writeln!(f, "  Iterations: {}", iters)?;
        }

        // Show robustness status
        if self.robustness_weights.is_some() {
            writeln!(f, "  Robustness: Applied")?;
        }

        if self.has_cv_scores() {
            if let Some(best_score) = self.best_cv_score() {
                writeln!(f, "  Best CV score: {}", best_score)?;
            }
        }
        writeln!(f)?;

        if let Some(diag) = &self.diagnostics {
            writeln!(f, "{}", diag)?;
        }

        writeln!(f, "Smoothed Data:")?;

        // Determine which columns to show
        let has_std_err = self.standard_errors.is_some();
        let has_conf = self.has_confidence_intervals();
        let has_pred = self.has_prediction_intervals();
        let has_resid = self.residuals.is_some();
        let has_weights = self.robustness_weights.is_some();

        // Build header
        write!(f, "{:>8} {:>12}", "X", "Y_smooth")?;
        if has_std_err {
            write!(f, " {:>12}", "Std_Err")?;
        }
        if has_conf {
            write!(f, " {:>12} {:>12}", "Conf_Lower", "Conf_Upper")?;
        }
        if has_pred {
            write!(f, " {:>12} {:>12}", "Pred_Lower", "Pred_Upper")?;
        }
        if has_resid {
            write!(f, " {:>12}", "Residual")?;
        }
        if has_weights {
            write!(f, " {:>10}", "Rob_Weight")?;
        }
        writeln!(f)?;

        // Separator line
        let line_width = 21
            + if has_std_err { 13 } else { 0 }
            + if has_conf { 26 } else { 0 }
            + if has_pred { 26 } else { 0 }
            + if has_resid { 13 } else { 0 }
            + if has_weights { 11 } else { 0 };
        writeln!(f, "{:-<width$}", "", width = line_width)?;

        // Data rows (show first 10 and last 10 if more than 20 points)
        let n = self.x.len();
        let show_all = n <= 20;
        let rows_to_show: Vec<usize> = if show_all {
            (0..n).collect()
        } else {
            (0..10).chain(n - 10..n).collect()
        };

        let mut prev_idx = 0;
        for (i, &idx) in rows_to_show.iter().enumerate() {
            // Add ellipsis if we skipped rows
            if i > 0 && idx != prev_idx + 1 {
                writeln!(f, "{:>8}", "...")?;
            }
            prev_idx = idx;

            write!(f, "{:>8.2} {:>12.6}", self.x[idx], self.y[idx])?;

            // Standard error
            if has_std_err {
                if let Some(se) = &self.standard_errors {
                    write!(f, " {:>12.6}", se[idx])?;
                }
            }

            // Confidence intervals
            if has_conf {
                if let (Some(lower), Some(upper)) = (&self.confidence_lower, &self.confidence_upper)
                {
                    write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
                }
            }

            // Prediction intervals
            if has_pred {
                if let (Some(lower), Some(upper)) = (&self.prediction_lower, &self.prediction_upper)
                {
                    write!(f, " {:>12.6} {:>12.6}", lower[idx], upper[idx])?;
                }
            }

            // Residuals
            if has_resid {
                if let Some(resid) = &self.residuals {
                    write!(f, " {:>12.6}", resid[idx])?;
                }
            }

            // Robustness weights
            if has_weights {
                if let Some(weights) = &self.robustness_weights {
                    write!(f, " {:>10.4}", weights[idx])?;
                }
            }

            writeln!(f)?;
        }

        Ok(())
    }
}
