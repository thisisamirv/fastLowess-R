//! Batch adapter for standard LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the batch (standard) execution adapter for LOWESS
//! smoothing. It handles complete datasets in memory with sequential processing,
//! making it suitable for small to medium-sized datasets where all data is
//! available upfront. The batch adapter is the simplest and most straightforward
//! way to use LOWESS.
//!
//! ## Design notes
//!
//! * Processes entire dataset in a single pass.
//! * Automatically sorts data by x-values and unsorts results.
//! * Delegates computation to the execution engine.
//! * Supports all LOWESS features: robustness, CV, intervals, diagnostics.
//! * Uses builder pattern for configuration.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ### Batch Processing
//! The batch adapter:
//! 1. Validates input data
//! 2. Sorts data by x-values (required for LOWESS)
//! 3. Executes LOWESS smoothing via the engine
//! 4. Computes optional outputs (diagnostics, intervals, residuals)
//! 5. Unsorts results to match original input order
//! 6. Packages everything into a `LowessResult`
//!
//! ### Builder Pattern
//! Configuration is done through `BatchLowessBuilder`:
//! * Fluent API for setting parameters
//! * Sensible defaults for all parameters
//! * Validation deferred until `fit()` is called
//!
//! ### Automatic Sorting
//! LOWESS requires sorted x-values. The batch adapter:
//! * Automatically sorts input data by x
//! * Tracks original indices
//! * Unsorts all outputs to match original order
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Cross-validation**: Automatic fraction selection
//! * **Auto-convergence**: Adaptive iteration count
//! * **Confidence intervals**: Uncertainty in fitted curve
//! * **Prediction intervals**: Uncertainty for new observations
//! * **Diagnostics**: RMSE, MAE, R^2, AIC, AICc
//! * **Residuals**: Differences between original and smoothed values
//! * **Robustness weights**: Final weights from iterative refinement
//!
//! ## Invariants
//!
//! * Input arrays x and y must have the same length.
//! * All values must be finite (no NaN or infinity).
//! * At least 2 data points are required.
//! * Fraction must be in (0, 1].
//! * Output order matches input order (automatic unsorting).
//!
//! ## Non-goals
//!
//! * This adapter does not handle streaming data (use streaming adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter does not provide parallel execution (single-threaded).
//! * This adapter does not handle missing values (NaN).
//!
//! ## Visibility
//!
//! The batch adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `BatchLowess` is possible but not
//! the primary interface.

use crate::algorithms::interpolation::calculate_delta;
use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{LowessConfig, LowessExecutor, SmoothPassFn};
use crate::engine::output::LowessResult;
use crate::engine::validator::Validator;
use crate::evaluation::cv::CVMethod;
use crate::evaluation::diagnostics::Diagnostics;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::kernel::WeightFunction;
use crate::primitives::errors::LowessError;
use crate::primitives::partition::BoundaryPolicy;
use crate::primitives::sorting::{sort_by_x, unsort};

use core::fmt::Debug;
use core::result::Result;
use num_traits::Float;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// ============================================================================
// Batch LOWESS Builder
// ============================================================================

/// Builder for batch LOWESS processor.
#[derive(Debug, Clone)]
pub struct BatchLowessBuilder<T: Float> {
    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Optimization delta
    pub delta: Option<T>,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Confidence/Prediction interval configuration
    pub interval_type: Option<IntervalMethod<T>>,

    /// Fractions for cross-validation
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation method
    pub cv_method: Option<CVMethod>,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Tolerance for auto-convergence
    pub auto_convergence: Option<T>,

    /// Whether to compute diagnostic statistics
    pub return_diagnostics: bool,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Policy for handling data boundaries
    pub boundary_policy: BoundaryPolicy,

    /// Optional custom smoothing function (e.g., for parallel execution)
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Tracks if any parameter was set multiple times (for validation)
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for BatchLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> BatchLowessBuilder<T> {
    /// Create a new batch LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap(),
            iterations: 3,
            delta: None,
            weight_function: WeightFunction::default(),
            robustness_method: RobustnessMethod::default(),
            interval_type: None,
            cv_fractions: None,
            cv_method: None,
            deferred_error: None,
            auto_convergence: None,
            return_diagnostics: false,
            compute_residuals: false,
            return_robustness_weights: false,
            zero_weight_fallback: ZeroWeightFallback::default(),
            boundary_policy: BoundaryPolicy::default(),
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

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = Some(delta);
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
    // Batch-Specific Setters
    // ========================================================================

    /// Enable returning diagnostics in the result.
    pub fn return_diagnostics(mut self, enabled: bool) -> Self {
        self.return_diagnostics = enabled;
        self
    }

    /// Enable confidence intervals at the specified level.
    pub fn confidence_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::confidence(level));
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        self.interval_type = Some(IntervalMethod::prediction(level));
        self
    }

    /// Enable cross-validation with the specified fractions.
    pub fn cross_validate(mut self, fractions: Vec<T>) -> Self {
        self.cv_fractions = Some(fractions);
        self
    }

    /// Set the cross-validation method.
    pub fn cv_method(mut self, method: CVMethod) -> Self {
        self.cv_method = Some(method);
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the batch processor.
    pub fn build(self) -> Result<BatchLowess<T>, LowessError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }

        // Check for duplicate parameter configuration
        Validator::validate_no_duplicates(self.duplicate_param)?;

        // Validate fraction
        Validator::validate_fraction(self.fraction)?;

        // Validate iterations
        Validator::validate_iterations(self.iterations)?;

        // Validate delta
        if let Some(delta) = self.delta {
            Validator::validate_delta(delta)?;
        }

        // Validate interval type
        if let Some(ref method) = self.interval_type {
            Validator::validate_interval_level(method.level)?;
        }

        // Validate CV fractions
        if let Some(ref fracs) = self.cv_fractions {
            Validator::validate_cv_fractions(fracs)?;
        }

        // Validate auto convergence tolerance
        if let Some(tol) = self.auto_convergence {
            Validator::validate_tolerance(tol)?;
        }

        Ok(BatchLowess { config: self })
    }
}

// ============================================================================
// Batch LOWESS Processor
// ============================================================================

/// Batch LOWESS processor.
pub struct BatchLowess<T: Float> {
    config: BatchLowessBuilder<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> BatchLowess<T> {
    /// Perform LOWESS smoothing on the provided data.
    pub fn fit(self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        Validator::validate_inputs(x, y)?;

        // Sort data by x using sorting module
        let sorted = sort_by_x(x, y);
        let delta = calculate_delta(self.config.delta, &sorted.x)?;

        let zw_flag: u8 = self.config.zero_weight_fallback.to_u8();

        // Configure batch execution
        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: zw_flag,
            robustness_method: self.config.robustness_method,
            cv_fractions: self.config.cv_fractions,
            cv_method: self.config.cv_method,
            auto_convergence: self.config.auto_convergence,
            return_variance: self.config.interval_type,
            boundary_policy: self.config.boundary_policy,
            custom_smooth_pass: self.config.custom_smooth_pass,
        };

        // Execute unified LOWESS
        let result = LowessExecutor::run_with_config(&sorted.x, &sorted.y, config);

        let y_smooth = result.smoothed;
        let std_errors = result.std_errors;
        let iterations_used = result.iterations;
        let fraction_used = result.used_fraction;
        let cv_scores = result.cv_scores;

        // Calculate residuals
        let n = x.len();
        let mut residuals = Vec::with_capacity(n);
        for (i, &smoothed_val) in y_smooth.iter().enumerate().take(n) {
            residuals.push(sorted.y[i] - smoothed_val);
        }

        // Get robustness weights from executor result (final iteration weights)
        let rob_weights = if self.config.return_robustness_weights {
            result.robustness_weights
        } else {
            Vec::new()
        };

        // Compute diagnostic statistics if requested
        let diagnostics = if self.config.return_diagnostics {
            Some(Diagnostics::compute(
                &sorted.y,
                &y_smooth,
                &residuals,
                std_errors.as_deref(),
            ))
        } else {
            None
        };

        // Compute intervals
        let (conf_lower, conf_upper, pred_lower, pred_upper) =
            if let Some(method) = &self.config.interval_type {
                if let Some(se) = &std_errors {
                    method.compute_intervals(&y_smooth, se, &residuals)?
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };

        // Unsort results using sorting module
        let indices = &sorted.indices;
        let y_smooth_out = unsort(&y_smooth, indices);
        let std_errors_out = std_errors.as_ref().map(|se| unsort(se, indices));
        let residuals_out = if self.config.compute_residuals {
            Some(unsort(&residuals, indices))
        } else {
            None
        };
        let rob_weights_out = if self.config.return_robustness_weights {
            Some(unsort(&rob_weights, indices))
        } else {
            None
        };
        let cl_out = conf_lower.as_ref().map(|v| unsort(v, indices));
        let cu_out = conf_upper.as_ref().map(|v| unsort(v, indices));
        let pl_out = pred_lower.as_ref().map(|v| unsort(v, indices));
        let pu_out = pred_upper.as_ref().map(|v| unsort(v, indices));

        Ok(LowessResult {
            x: x.to_vec(),
            y: y_smooth_out,
            standard_errors: std_errors_out,
            confidence_lower: cl_out,
            confidence_upper: cu_out,
            prediction_lower: pl_out,
            prediction_upper: pu_out,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            fraction_used,
            iterations_used,
            cv_scores,
            diagnostics,
        })
    }
}
