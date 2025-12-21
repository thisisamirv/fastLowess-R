//! Robustness weight computation for outlier downweighting.
//!
//! ## Purpose
//!
//! This module implements iterative reweighted least squares (IRLS) for robust
//! LOWESS smoothing. After an initial fit, residuals are computed and used to
//! downweight outliers in subsequent iterations. This makes LOWESS resistant
//! to outliers and provides more reliable smoothing in the presence of
//! contaminated data.
//!
//! ## Design notes
//!
//! * Uses MAD (Median Absolute Deviation) for robust scale estimation.
//! * Falls back to MAR (Mean Absolute Residual) when MAD is too small.
//! * Provides three robustness methods: Bisquare, Huber, and Talwar.
//! * Bisquare (Tukey's biweight) is the default, following Cleveland (1979).
//! * All computations are generic over `Float` types to support f32 and f64.
//! * Tuning constants are method-specific and based on statistical literature.
//!
//! ## Available methods
//!
//! * **Bisquare** (default): Tukey's bisquare (biweight) function (c=6.0)
//! * **Huber**: Huber M-estimator weights (c=1.345)
//! * **Talwar**: Hard threshold - complete rejection beyond threshold (c=2.5)
//!
//! ## Key concepts
//!
//! ### Iterative Reweighted Least Squares (IRLS)
//! For each robustness iteration:
//! 1. Compute residuals: r_i = y_i - ŷ_i
//! 2. Estimate robust scale: s = MAD(r) (or fallback to MAR)
//! 3. Normalize residuals: u_i = r_i / (c × s)
//! 4. Apply weight function: w_i = ρ(u_i)
//! 5. Re-fit using combined weights: kernel × robustness
//!
//! ### Bisquare Weights
//! Tukey's bisquare function provides smooth downweighting:
//!
//! w(u) = (1 - u^2)^2  if |u| < 1
//!
//! w(u) = 0          if |u| >= 1
//!
//! ### Huber Weights
//! Huber weights provide less aggressive downweighting:
//!
//! w(u) = 1      if |u| <= c
//!
//! w(u) = c / |u|  if |u| > c
//!
//! ### Talwar Weights
//! Hard threshold - most aggressive rejection:
//!
//! w(u) = 1  if |u| <= c
//!
//! w(u) = 0  if |u| > c
//!
//! ### Robust Scale Estimation
//! Uses MAD (Median Absolute Deviation) to estimate the scale of residuals.
//! Falls back to MAR (Mean Absolute Residual) when MAD is too small relative
//! to MAR, ensuring numerical stability even with highly clustered residuals.
//!
//! ## Invariants
//!
//! * Robustness weights are in [0, 1].
//! * Scale estimates are always positive.
//! * Tuning constants are method-specific and positive.
//! * Weight functions are continuous (except Talwar).
//!
//! ## Non-goals
//!
//! * This module does not perform the regression itself.
//! * This module does not compute residuals (done by fitting algorithm).
//! * This module does not decide the number of robustness iterations.
//! * This module does not provide adaptive tuning constant selection.
//!
//! ## Visibility
//!
//! The [`RobustnessMethod`] enum is part of the public API, allowing users
//! to select the robustness method. Internal implementation details may
//! change without notice.

#[cfg(not(feature = "std"))]
use num_traits::Float;
#[cfg(feature = "std")]
use num_traits::Float;

use crate::math::mad::compute_mad;

// ============================================================================
// Robustness Method
// ============================================================================

/// Robustness weighting method for outlier downweighting.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustnessMethod {
    /// Bisquare (Tukey's biweight) - default and most common.
    ///
    /// Uses tuning constant c=6.0 following Cleveland (1979).
    /// Provides smooth downweighting with complete rejection beyond threshold.
    #[default]
    Bisquare,

    /// Huber weights - less aggressive downweighting.
    ///
    /// Uses tuning constant c=1.345 for 95% efficiency at the normal distribution.
    /// Never completely rejects points, only downweights them.
    Huber,

    /// Talwar (hard threshold) - most aggressive.
    ///
    /// Uses tuning constant c=2.5.
    /// Completely rejects points beyond threshold (weight = 0).
    Talwar,
}

// ============================================================================
// Implementation
// ============================================================================

impl RobustnessMethod {
    // ========================================================================
    // Constants
    // ========================================================================

    /// Default tuning constant for bisquare robustness weights.
    ///
    /// Value of 6.0 follows Cleveland (1979) and is applied to the raw MAD.
    const DEFAULT_BISQUARE_C: f64 = 6.0;

    /// Default tuning constant for Huber weights.
    ///
    /// Value of 1.345 is the standard threshold for 95% efficiency.
    /// Note: This is applied directly to the MAD-scaled residuals.
    const DEFAULT_HUBER_C: f64 = 1.345;

    /// Default tuning constant for Talwar weights.
    ///
    /// Value of 2.5 provides aggressive outlier rejection.
    const DEFAULT_TALWAR_C: f64 = 2.5;

    /// Minimum scale threshold relative to mean absolute residual.
    ///
    /// If MAD < SCALE_THRESHOLD × MAR, use MAR instead of MAD.
    const SCALE_THRESHOLD: f64 = 1e-7;

    /// Minimum tuned-scale absolute epsilon to avoid division by zero.
    const MIN_TUNED_SCALE: f64 = 1e-12;

    // ========================================================================
    // Main API
    // ========================================================================

    /// Apply robustness weights using the configured method.
    pub fn apply_robustness_weights<T: Float>(&self, residuals: &[T], weights: &mut [T]) {
        if residuals.is_empty() {
            return;
        }

        let base_scale = self.compute_scale(residuals);

        let (method_type, tuning_constant) = match self {
            Self::Bisquare => (0, Self::DEFAULT_BISQUARE_C),
            Self::Huber => (1, Self::DEFAULT_HUBER_C),
            Self::Talwar => (2, Self::DEFAULT_TALWAR_C),
        };

        let c_t = T::from(tuning_constant).unwrap();

        for (i, &r) in residuals.iter().enumerate() {
            weights[i] = match method_type {
                0 => Self::bisquare_weight(r, base_scale, c_t),
                1 => Self::huber_weight(r, base_scale, c_t),
                _ => Self::talwar_weight(r, base_scale, c_t),
            };
        }
    }

    // ========================================================================
    // Scale Estimation
    // ========================================================================

    /// Compute robust scale estimate with MAD fallback.
    fn compute_scale<T: Float>(&self, residuals: &[T]) -> T {
        let mad = compute_mad(residuals);

        // Compute MAR (Mean Absolute Residual) inline
        let mean_abs = if residuals.is_empty() {
            T::zero()
        } else {
            let sum_abs = residuals.iter().fold(T::zero(), |acc, &r| acc + r.abs());
            sum_abs / T::from(residuals.len()).unwrap()
        };

        let relative_threshold = T::from(Self::SCALE_THRESHOLD).unwrap() * mean_abs;
        let absolute_threshold = T::from(Self::MIN_TUNED_SCALE).unwrap();
        let scale_threshold = relative_threshold.max(absolute_threshold);

        if mad <= scale_threshold {
            // MAD is too small, use MAR as fallback
            mean_abs.max(mad)
        } else {
            mad
        }
    }

    // ========================================================================
    // Weight Functions
    // ========================================================================

    /// Compute bisquare weight.
    ///
    /// # Formula
    ///
    /// u = |r| / (c * s)
    ///
    /// w(u) = (1 - u^2)^2  if u < 1
    ///
    /// w(u) = 0          if u >= 1
    #[inline]
    fn bisquare_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let min_eps = T::from(Self::MIN_TUNED_SCALE).unwrap();
        // Ensure c is at least min_eps so tuned_scale isn't zero
        let c_clamped = c.max(min_eps);
        let tuned_scale = (scale * c_clamped).max(min_eps);

        let u = (residual / tuned_scale).abs();
        if u >= T::one() {
            T::zero()
        } else {
            let tmp = T::one() - u * u;
            tmp * tmp
        }
    }

    /// Compute Huber weight.
    ///
    /// # Formula
    ///
    /// u = |r| / s
    ///
    /// w(u) = 1      if u <= c
    ///
    /// w(u) = c / u  if u > c
    #[inline]
    fn huber_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { c / u }
    }

    /// Compute Talwar weight.
    ///
    /// # Formula
    ///
    /// u = |r| / s
    ///
    /// w(u) = 1  if u <= c
    ///
    /// w(u) = 0  if u > c
    #[inline]
    fn talwar_weight<T: Float>(residual: T, scale: T, c: T) -> T {
        if scale <= T::zero() {
            return T::one();
        }

        let u = (residual / scale).abs();
        if u <= c { T::one() } else { T::zero() }
    }
}
