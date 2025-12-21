//! Online adapter for incremental LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the online (incremental) execution adapter for LOWESS
//! smoothing. It maintains a sliding window of recent observations and produces
//! smoothed values for new points as they arrive, making it suitable for
//! real-time data streams and incremental updates where data arrives
//! sequentially.
//!
//! ## Design notes
//!
//! * Uses a fixed-size circular buffer (VecDeque) for the sliding window.
//! * Automatically evicts oldest points when capacity is reached.
//! * Performs full LOWESS smoothing on the current window for each new point.
//! * Supports basic smoothing and residuals only (no intervals or diagnostics).
//! * Requires minimum number of points before smoothing begins.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ### Sliding Window
//! The online adapter maintains a fixed-size window:
//! ```text
//! Initial state (capacity=5):
//! Buffer: [_, _, _, _, _]
//!
//! After 3 points:
//! Buffer: [x1, x2, x3, _, _]
//!
//! After 7 points (buffer full, oldest dropped):
//! Buffer: [x3, x4, x5, x6, x7]
//!         ↑ oldest    newest ↑
//! ```
//!
//! ### Incremental Processing
//! For each new point:
//! 1. Validate the point (finite values)
//! 2. Add to window (evict oldest if at capacity)
//! 3. Check if minimum points threshold is met
//! 4. Perform LOWESS smoothing on current window
//! 5. Return smoothed value for the newest point
//!
//! ### Update Modes
//! * **Incremental**: Basic incremental updates (currently performs full re-smooth)
//! * **Full**: Full re-smooth of the current window (O(window^2))
//!
//! ### Initialization Phase
//! Before `min_points` are accumulated, `add_point()` returns `None`.
//! Once enough points are available, smoothing begins.
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Residuals**: Differences between original and smoothed values
//! * **Window snapshots**: Get full `LowessResult` for current window
//! * **Reset capability**: Clear window for handling data gaps
//!
//! ## Invariants
//!
//! * Window size never exceeds capacity.
//! * All values in window are finite (no NaN or infinity).
//! * At least `min_points` are required before smoothing.
//! * Window maintains insertion order (oldest to newest).
//! * Smoothing is performed on sorted window data.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not compute diagnostic statistics.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing (use batch adapter).
//! * This adapter does not handle out-of-order points.
//!
//! ## Visibility
//!
//! The online adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `OnlineLowess` is possible but not
//! the primary interface.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::VecDeque, vec::Vec};

#[cfg(feature = "std")]
use std::{collections::VecDeque, vec::Vec};

use crate::algorithms::regression::{LinearRegression, ZeroWeightFallback};
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{LowessConfig, LowessExecutor, SmoothPassFn};
use crate::engine::validator::Validator;
use crate::math::kernel::WeightFunction;
use crate::primitives::errors::LowessError;
use crate::primitives::partition::{BoundaryPolicy, UpdateMode};

use core::fmt::Debug;
use core::result::Result;
use num_traits::Float;

// ============================================================================
// Online LOWESS Builder
// ============================================================================

/// Builder for online LOWESS processor.
#[derive(Debug, Clone)]
pub struct OnlineLowessBuilder<T: Float> {
    /// Window capacity (maximum number of points to retain)
    pub window_capacity: usize,

    /// Minimum points before smoothing starts
    pub min_points: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Delta parameter for interpolation optimization
    pub delta: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Convergence tolerance for early stopping (None = disabled)
    pub auto_convergence: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Update mode for incremental processing
    pub update_mode: UpdateMode,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Optional custom smoothing function (e.g., for parallel execution)
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Tracks if any parameter was set multiple times (for validation)
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for OnlineLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> OnlineLowessBuilder<T> {
    /// Create a new online LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            window_capacity: 1000,
            min_points: 3,
            fraction: T::from(0.2).unwrap(),
            delta: T::zero(),
            iterations: 1,
            weight_function: WeightFunction::default(),
            update_mode: UpdateMode::default(),
            robustness_method: RobustnessMethod::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
            compute_residuals: false,
            return_robustness_weights: false,
            auto_convergence: None,
            deferred_error: None,
            custom_smooth_pass: None,
            duplicate_param: None,
        }
    }

    // ========================================================================
    // Shared Setters
    // ========================================================================

    /// Set the smoothing fraction (span).
    pub fn fraction(mut self, fraction: T) -> Self {
        self.fraction = fraction;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set the delta parameter for interpolation-based optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the robustness method for outlier handling.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set the zero-weight fallback policy.
    pub fn zero_weight_fallback(mut self, fallback: ZeroWeightFallback) -> Self {
        self.zero_weight_fallback = fallback;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    /// Enable auto-convergence for robustness iterations.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        self.auto_convergence = Some(tolerance);
        self
    }

    /// Enable returning residuals in the output.
    pub fn compute_residuals(mut self, enabled: bool) -> Self {
        self.compute_residuals = enabled;
        self
    }

    /// Enable returning robustness weights in the result.
    pub fn return_robustness_weights(mut self, enabled: bool) -> Self {
        self.return_robustness_weights = enabled;
        self
    }

    // ========================================================================
    // Online-Specific Setters
    // ========================================================================

    /// Set window capacity (maximum number of points to retain).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        self.window_capacity = capacity;
        self
    }

    /// Set minimum points before smoothing starts.
    pub fn min_points(mut self, min: usize) -> Self {
        self.min_points = min;
        self
    }

    /// Set the update mode for incremental processing.
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        self.update_mode = mode;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the online processor.
    pub fn build(self) -> Result<OnlineLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate configuration early
        Validator::validate_window_capacity(self.window_capacity, 3)?;
        Validator::validate_min_points(self.min_points, self.window_capacity)?;

        let capacity = self.window_capacity;
        Ok(OnlineLowess {
            config: self,
            window_x: VecDeque::with_capacity(capacity),
            window_y: VecDeque::with_capacity(capacity),
        })
    }
}

// ============================================================================
// Online LOWESS Output
// ============================================================================

/// Result of a single online update.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineOutput<T> {
    /// Smoothed value for the latest point
    pub smoothed: T,

    /// Standard error (if computed)
    pub std_error: Option<T>,

    /// Residual (y - smoothed)
    pub residual: Option<T>,

    /// Robustness weight for the latest point (if computed)
    pub robustness_weight: Option<T>,
}

// ============================================================================
// Online LOWESS Processor
// ============================================================================

/// Online LOWESS processor for streaming data.
pub struct OnlineLowess<T: Float> {
    config: OnlineLowessBuilder<T>,
    window_x: VecDeque<T>,
    window_y: VecDeque<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> OnlineLowess<T> {
    /// Add a new point and get its smoothed value.
    pub fn add_point(&mut self, x: T, y: T) -> Result<Option<OnlineOutput<T>>, LowessError> {
        // Validate new point
        if !x.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "x={}",
                x.to_f64().unwrap_or(f64::NAN)
            )));
        }
        if !y.is_finite() {
            return Err(LowessError::InvalidNumericValue(format!(
                "y={}",
                y.to_f64().unwrap_or(f64::NAN)
            )));
        }

        // Add to window
        self.window_x.push_back(x);
        self.window_y.push_back(y);

        // Evict oldest if over capacity
        if self.window_x.len() > self.config.window_capacity {
            self.window_x.pop_front();
            self.window_y.pop_front();
        }

        // Check if we have enough points
        if self.window_x.len() < self.config.min_points {
            return Ok(None);
        }

        // Convert window to vectors for smoothing
        let x_vec: Vec<T> = self.window_x.iter().copied().collect();
        let y_vec: Vec<T> = self.window_y.iter().copied().collect();

        // Special case: exactly two points, use exact linear fit
        if x_vec.len() == 2 {
            let x0 = x_vec[0];
            let x1 = x_vec[1];
            let y0 = y_vec[0];
            let y1 = y_vec[1];

            let smoothed = if x1 != x0 {
                let slope = (y1 - y0) / (x1 - x0);
                y0 + slope * (x1 - x0)
            } else {
                // Identical x: use mean for stability
                (y0 + y1) / T::from(2.0).unwrap()
            };

            let residual = y - smoothed;

            return Ok(Some(OnlineOutput {
                smoothed,
                std_error: None,
                residual: Some(residual),
                robustness_weight: Some(T::one()),
            }));
        }

        // Smooth using LOWESS for windows of size >= 3
        let zero_flag = self.config.zero_weight_fallback.to_u8();

        // Choose update strategy based on configuration
        let (smoothed, std_err, rob_weight) = match self.config.update_mode {
            UpdateMode::Incremental => {
                // Incremental mode: fit only the latest point
                let n = x_vec.len();
                let window_size = (self.config.fraction * T::from(n).unwrap())
                    .ceil()
                    .to_usize()
                    .unwrap_or(n)
                    .max(2)
                    .min(n);

                let fitter = LinearRegression;
                let mut weights = vec![T::zero(); n];
                let robustness_weights = vec![T::one(); n];

                let (smoothed_val, _) = LowessExecutor::fit_single_point(
                    &x_vec,
                    &y_vec,
                    n - 1, // Latest point
                    window_size,
                    false, // No robustness for single point
                    &robustness_weights,
                    &mut weights,
                    self.config.weight_function,
                    self.config.zero_weight_fallback,
                    &fitter,
                );

                (smoothed_val, None, Some(T::one()))
            }
            UpdateMode::Full => {
                // Full mode: re-smooth entire window
                let config = LowessConfig {
                    fraction: Some(self.config.fraction),
                    iterations: self.config.iterations,
                    delta: self.config.delta,
                    weight_function: self.config.weight_function,
                    robustness_method: self.config.robustness_method,
                    zero_weight_fallback: zero_flag,
                    boundary_policy: self.config.boundary_policy,
                    custom_smooth_pass: self.config.custom_smooth_pass,
                    auto_convergence: self.config.auto_convergence,
                    cv_fractions: None,
                    cv_method: None,
                    return_variance: None,
                };

                let result = LowessExecutor::run_with_config(&x_vec, &y_vec, config.clone());
                let smoothed_vec = result.smoothed;
                let se_vec = result.std_errors;

                let smoothed_val = smoothed_vec.last().copied().ok_or_else(|| {
                    LowessError::InvalidNumericValue("No smoothed output produced".into())
                })?;
                let std_err = se_vec.as_ref().and_then(|v| v.last().copied());
                let rob_weight = if self.config.return_robustness_weights {
                    result.robustness_weights.last().copied()
                } else {
                    None
                };

                (smoothed_val, std_err, rob_weight)
            }
        };

        let residual = y - smoothed;

        Ok(Some(OnlineOutput {
            smoothed,
            std_error: std_err,
            residual: Some(residual),
            robustness_weight: rob_weight,
        }))
    }

    /// Get the current window size.
    pub fn window_size(&self) -> usize {
        self.window_x.len()
    }

    /// Clear the window.
    pub fn reset(&mut self) {
        self.window_x.clear();
        self.window_y.clear();
    }
}
