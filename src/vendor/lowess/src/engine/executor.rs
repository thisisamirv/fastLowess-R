//! Execution engine for LOWESS smoothing operations.
//!
//! ## Purpose
//!
//! This module provides the core execution engine that orchestrates LOWESS
//! smoothing operations. It handles the iteration loop, robustness weight
//! updates, convergence checking, cross-validation, and variance estimation.
//! The executor is the central component that coordinates all lower-level
//! algorithms to produce smoothed results.
//!
//! ## Design notes
//!
//! * Provides both configuration-based and parameter-based entry points.
//! * Handles cross-validation for automatic fraction selection.
//! * Supports auto-convergence for adaptive iteration counts.
//! * Manages working buffers efficiently to minimize allocations.
//! * Uses delta optimization for performance on dense data.
//! * Separates concerns: fitting, interpolation, robustness, convergence.
//! * Generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ## Execution Flow
//!
//! 1. Validate and prepare parameters (window size, delta, etc.)
//! 2. Allocate [`IterationBuffers`] (y_smooth, weights, residuals)
//! 3. Perform initial smoothing pass (iteration 0)
//! 4. For each robustness iteration:
//!    - Compute residuals: r_i = y_i - y_hat_i
//!    - Update robustness weights using configured [`RobustnessMethod`]
//!    - Re-smooth with combined weights (kernel * robustness)
//!    - Check convergence (if enabled)
//! 5. Optionally compute standard errors via [`IntervalMethod`]
//! 6. Return [`ExecutorOutput`]
//!
//! ### Delta Optimization
//!
//! Instead of fitting every point, the executor uses an optimization for dense data:
//! * Selects anchor points based on the `delta` threshold
//! * Explicitly fits these anchor points using local regression
//! * Linearly interpolates between anchors for intermediate points
//!
//! This provides significant speedup on dense data with minimal accuracy loss,
//! following the approach described in Cleveland (1979).
//!
//! ### Auto-Convergence
//!
//! When enabled, robustness iterations stop early if the maximum absolute
//! change in smoothed values falls below the tolerance threshold:
//! max |y_hat_i^(k) - y_hat_i^(k-1)| < tolerance.
//!
//! ### Cross-Validation
//!
//! When `cv_fractions` are provided, the executor:
//! * Tests each fraction using the specified [`CVMethod`] (K-Fold or LOOCV)
//! * Selects the fraction with minimum Root Mean Squared Error (RMSE)
//! * Performs final smoothing with the optimal fraction
//!
//! ## Invariants
//!
//! * Input x-values are assumed to be monotonically increasing (sorted).
//! * All working buffers have the same length as input data.
//! * Robustness weights are always in [0, 1].
//! * Window size is at least 2 and at most n.
//! * Iteration count is non-negative.
//!
//! ## Non-goals
//!
//! * This module does not validate input data (handled by `validator`).
//! * This module does not sort input data (caller's responsibility).
//! * This module does not provide public-facing result formatting.
//! * This module does not handle parallel execution directly (handled by adapters).
//!
//! ## Visibility
//!
//! This module is an internal implementation detail used by the LOWESS
//! adapters. The [`LowessExecutor`] struct and [`LowessConfig`] are intended
//! for internal use and may change without notice.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::algorithms::interpolation::interpolate_gap;
use crate::algorithms::regression::{
    GLSModel, LinearRegression, Regression, RegressionContext, ZeroWeightFallback,
};
use crate::algorithms::robustness::RobustnessMethod;
use crate::evaluation::cv::CVMethod;
use crate::evaluation::intervals::IntervalMethod;
use crate::math::boundary::apply_boundary_policy;
use crate::math::kernel::WeightFunction;
use crate::primitives::partition::BoundaryPolicy;
use crate::primitives::window::Window;

use core::fmt::Debug;
use core::mem;
use num_traits::Float;

// ============================================================================
// Output Types
// ============================================================================

/// Output from LOWESS execution.
#[derive(Debug, Clone)]
pub struct ExecutorOutput<T> {
    /// Smoothed y-values.
    pub smoothed: Vec<T>,

    /// Standard errors (if SE estimation or intervals were requested).
    pub std_errors: Option<Vec<T>>,

    /// Number of iterations performed (if auto-convergence was active).
    pub iterations: Option<usize>,

    /// Smoothing fraction used (selected by CV or configured).
    pub used_fraction: T,

    /// RMSE scores for each tested fraction (if CV was performed).
    pub cv_scores: Option<Vec<T>>,

    /// Final robustness weights from iterative refinement.
    pub robustness_weights: Vec<T>,
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for LOWESS execution.
#[derive(Debug, Clone)]
pub struct LowessConfig<T> {
    /// Smoothing fraction (0, 1].
    /// If `None` and `cv_fractions` are provided, bandwidth selection is performed.
    pub fraction: Option<T>,

    /// Number of robustness iterations (0 means initial fit only).
    pub iterations: usize,

    /// Delta parameter for linear interpolation optimization.
    pub delta: T,

    /// Kernel weight function used for local regression.
    pub weight_function: WeightFunction,

    /// Zero-weight fallback policy (via [`ZeroWeightFallback`]).
    pub zero_weight_fallback: u8,

    /// Robustness weighting method for outlier downweighting.
    pub robustness_method: RobustnessMethod,

    /// Candidate fractions to evaluate during cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    /// Cross-validation strategy (e.g., K-Fold or LOOCV).
    pub cv_method: Option<CVMethod>,

    /// Convergence tolerance for early stopping of robustness iterations.
    pub auto_convergence: Option<T>,

    /// Configuration for standard errors and intervals.
    pub return_variance: Option<IntervalMethod<T>>,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Custom smooth pass function (enables parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,
}

impl<T: Float> Default for LowessConfig<T> {
    fn default() -> Self {
        Self {
            fraction: None,
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::default(),
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            return_variance: None,
            boundary_policy: BoundaryPolicy::default(),
            custom_smooth_pass: None,
        }
    }
}

// ============================================================================
// Internal Types
// ============================================================================

/// Working buffers for LOWESS iteration.
pub struct IterationBuffers<T> {
    /// Current smoothed values
    pub y_smooth: Vec<T>,

    /// Previous iteration values (for convergence check)
    pub y_prev: Vec<T>,

    /// Robustness weights
    pub robustness_weights: Vec<T>,

    /// Residuals buffer
    pub residuals: Vec<T>,

    /// Kernel weights scratch buffer
    pub weights: Vec<T>,
}

impl<T: Float> IterationBuffers<T> {
    /// Allocate all working buffers for LOWESS iteration.
    pub fn allocate(n: usize, use_convergence: bool) -> Self {
        Self {
            y_smooth: vec![T::zero(); n],
            y_prev: if use_convergence {
                vec![T::zero(); n]
            } else {
                Vec::new()
            },
            robustness_weights: vec![T::one(); n],
            residuals: vec![T::zero(); n],
            weights: vec![T::zero(); n],
        }
    }
}

// ============================================================================
// LowessExecutor
// ============================================================================

/// Signature for custom smooth pass function
pub type SmoothPassFn<T> = fn(
    &[T],           // x
    &[T],           // y
    usize,          // window_size
    T,              // delta (interpolation optimization threshold)
    bool,           // use_robustness
    &[T],           // robustness_weights
    &mut [T],       // output (y_smooth)
    WeightFunction, // weight_function
    u8,             // zero_weight_flag
);

/// Unified executor for LOWESS smoothing operations.
#[derive(Debug, Clone)]
pub struct LowessExecutor<T: Float> {
    /// Smoothing fraction (0, 1].
    pub fraction: T,

    /// Number of robustness iterations.
    pub iterations: usize,

    /// Delta for interpolation optimization.
    pub delta: T,

    /// Kernel weight function.
    pub weight_function: WeightFunction,

    /// Zero weight fallback flag (0=UseLocalMean, 1=ReturnOriginal, 2=ReturnNone).
    pub zero_weight_fallback: u8,

    /// Robustness method for iterative refinement.
    pub robustness_method: RobustnessMethod,

    /// Boundary handling policy.
    pub boundary_policy: BoundaryPolicy,

    /// Custom smooth pass function (e.g., for parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,
}

impl<T: Float> Default for LowessExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessExecutor<T> {
    // ========================================================================
    // Constructor and Builder Methods
    // ========================================================================

    /// Create a new executor with default parameters.
    pub fn new() -> Self {
        Self {
            fraction: T::from(0.67).unwrap_or_else(|| T::from(0.5).unwrap()),
            iterations: 3,
            delta: T::zero(),
            weight_function: WeightFunction::Tricube,
            zero_weight_fallback: 0,
            robustness_method: RobustnessMethod::Bisquare,
            boundary_policy: BoundaryPolicy::default(),
            custom_smooth_pass: None,
        }
    }

    /// Set the smoothing fraction (bandwidth).
    pub fn fraction(mut self, frac: T) -> Self {
        self.fraction = frac;
        self
    }

    /// Set the number of robustness iterations.
    pub fn iterations(mut self, niter: usize) -> Self {
        self.iterations = niter;
        self
    }

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        self.weight_function = wf;
        self
    }

    /// Set the zero weight fallback policy flag.
    pub fn zero_weight_fallback(mut self, flag: u8) -> Self {
        self.zero_weight_fallback = flag;
        self
    }

    /// Set the robustness method for iterative refinement.
    pub fn robustness_method(mut self, method: RobustnessMethod) -> Self {
        self.robustness_method = method;
        self
    }

    /// Set a custom smooth pass function (e.g., for parallelization).
    pub fn custom_smooth_pass(mut self, smooth_pass_fn: Option<SmoothPassFn<T>>) -> Self {
        self.custom_smooth_pass = smooth_pass_fn;
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary_policy = policy;
        self
    }

    // ========================================================================
    // Specialized Fitting Functions
    // ========================================================================

    /// Fit the first point and initialize the smoothing window.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_single_point<Fitter>(
        x: &[T],
        y: &[T],
        idx: usize,
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
    ) -> (T, Window)
    where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut window = Window::initialize(idx, window_size, n);
        window.recenter(x, idx, n);

        let ctx = RegressionContext {
            x,
            y,
            idx,
            window,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
        };

        (fitter.fit(ctx).unwrap_or_else(|| y[idx]), window)
    }

    /// Fit the first point and initialize the smoothing window.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_first_point<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
    ) -> Window
    where
        Fitter: Regression<T> + ?Sized,
    {
        let (val, window) = Self::fit_single_point(
            x,
            y,
            0,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
        );
        y_smooth[0] = val;
        window
    }

    /// Main fitting loop: iterate through remaining points with delta-skipping
    /// and linear interpolation.
    #[allow(clippy::too_many_arguments)]
    fn fit_and_interpolate_remaining<Fitter>(
        x: &[T],
        y: &[T],
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        weights: &mut [T],
        weight_function: WeightFunction,
        zero_weight_fallback: ZeroWeightFallback,
        fitter: &Fitter,
        y_smooth: &mut [T],
        mut window: Window,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        if n <= 1 {
            return;
        }

        let mut last_fitted = 0usize;
        let mut k = 1usize;

        // Main loop: process all points
        while k < n {
            // Inline delta skip logic (matches statsmodels for performance)
            // Simple forward linear scan - cache friendly, minimal branching
            let cutpoint = x[last_fitted] + delta;

            // Scan forward, handling tied values inline
            while k < n && x[k] <= cutpoint {
                // Handle tied x-values: copy fitted value
                if x[k] == x[last_fitted] {
                    y_smooth[k] = y_smooth[last_fitted];
                    last_fitted = k;
                }
                k += 1;
            }

            // Determine current anchor point to fit
            // Either k-1 (last point within delta) or at minimum last_fitted+1
            let current = if k > 0 {
                usize::max(k.saturating_sub(1), last_fitted + 1).min(n - 1)
            } else {
                (last_fitted + 1).min(n - 1)
            };

            // Check if we've made progress
            if current <= last_fitted {
                break;
            }

            // Update window to be centered around current point
            window.recenter(x, current, n);

            // Fit current point
            let ctx = RegressionContext {
                x,
                y,
                idx: current,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
            };

            y_smooth[current] = fitter.fit(ctx).unwrap_or_else(|| y[current]);

            // Linearly interpolate between last fitted and current
            interpolate_gap(x, y_smooth, last_fitted, current);
            last_fitted = current;
            k = current + 1;
        }

        // Final interpolation to the end if necessary
        if last_fitted < n.saturating_sub(1) {
            // Fit the last point explicitly
            let final_idx = n - 1;
            window.recenter(x, final_idx, n);

            let ctx = RegressionContext {
                x,
                y,
                idx: final_idx,
                window,
                use_robustness,
                robustness_weights,
                weights,
                weight_function,
                zero_weight_fallback,
            };

            y_smooth[final_idx] = fitter.fit(ctx).unwrap_or_else(|| y[final_idx]);
            interpolate_gap(x, y_smooth, last_fitted, final_idx);
        }
    }

    /// Perform a single smoothing pass over all points.
    #[allow(clippy::too_many_arguments)]
    pub fn smooth_pass<Fitter>(
        x: &[T],
        y: &[T],
        window_size: usize,
        delta: T,
        use_robustness: bool,
        robustness_weights: &[T],
        y_smooth: &mut [T],
        weight_function: WeightFunction,
        weights: &mut [T],
        zero_weight_flag: u8,
        fitter: &Fitter,
    ) where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        if n == 0 {
            return;
        }

        let zero_weight_fallback = ZeroWeightFallback::from_u8(zero_weight_flag);

        // Fit first point
        let window = Self::fit_first_point(
            x,
            y,
            window_size,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
        );

        // Fit remaining points with interpolation
        Self::fit_and_interpolate_remaining(
            x,
            y,
            delta,
            use_robustness,
            robustness_weights,
            weights,
            weight_function,
            zero_weight_fallback,
            fitter,
            y_smooth,
            window,
        );
    }

    // ========================================================================
    // Iteration Control Functions
    // ========================================================================

    /// Check convergence between current and previous smoothed values.
    pub fn check_convergence(y_smooth: &[T], y_prev: &[T], tolerance: T) -> bool {
        let max_change = y_smooth
            .iter()
            .zip(y_prev.iter())
            .fold(T::zero(), |maxv, (&current, &previous)| {
                T::max(maxv, (current - previous).abs())
            });

        max_change <= tolerance
    }

    /// Update robustness weights based on residuals.
    pub fn update_robustness_weights(
        y: &[T],
        y_smooth: &[T],
        residuals: &mut [T],
        robustness_weights: &mut [T],
        robustness_updater: &RobustnessMethod,
    ) {
        // Inline compute_residuals: residuals[i] = y[i] - y_smooth[i]
        for i in 0..y.len() {
            residuals[i] = y[i] - y_smooth[i];
        }
        robustness_updater.apply_robustness_weights(residuals, robustness_weights);
    }

    /// Compute standard errors for smoothed values.
    pub fn compute_std_errors(
        x: &[T],
        y: &[T],
        y_smooth: &[T],
        window_size: usize,
        robustness_weights: &[T],
        weight_function: WeightFunction,
        interval_method: &IntervalMethod<T>,
    ) -> Vec<T> {
        let n = x.len();
        let mut se = vec![T::zero(); n];
        interval_method.compute_window_se(
            x,
            y,
            y_smooth,
            window_size,
            robustness_weights,
            &mut se,
            &|t| weight_function.compute_weight(t),
        );
        se
    }

    /// Perform the full LOWESS iteration loop.
    #[allow(clippy::too_many_arguments)]
    pub fn iteration_loop_with_callback<Fitter>(
        &self,
        x: &[T],
        y: &[T],
        window_size: usize,
        niter: usize,
        delta: T,
        weight_function: WeightFunction,
        zero_weight_flag: u8,
        fitter: &Fitter,
        robustness_updater: &RobustnessMethod,
        interval_method: Option<&IntervalMethod<T>>,
        convergence_tolerance: Option<T>,
        smooth_pass_fn: Option<SmoothPassFn<T>>,
    ) -> (Vec<T>, Option<Vec<T>>, usize, Vec<T>)
    where
        Fitter: Regression<T> + ?Sized,
    {
        let n = x.len();
        let mut buffers = IterationBuffers::allocate(n, convergence_tolerance.is_some());
        let mut iterations_performed = 0;

        // Copy initial y values to y_smooth
        buffers.y_smooth.copy_from_slice(y);

        // Smoothing iterations with robustness updates
        for iter in 0..=niter {
            iterations_performed = iter;

            // Swap buffers if checking convergence (save previous state)
            if convergence_tolerance.is_some() && iter > 0 {
                mem::swap(&mut buffers.y_smooth, &mut buffers.y_prev);
            }

            // Perform smoothing pass
            if let Some(callback) = smooth_pass_fn {
                callback(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    zero_weight_flag,
                );
            } else {
                Self::smooth_pass(
                    x,
                    y,
                    window_size,
                    delta,
                    iter > 0, // use_robustness
                    &buffers.robustness_weights,
                    &mut buffers.y_smooth,
                    weight_function,
                    &mut buffers.weights,
                    zero_weight_flag,
                    fitter,
                );
            }

            // Check convergence if tolerance is provided (skip on first iteration)
            if let Some(tol) = convergence_tolerance {
                if iter > 0 && Self::check_convergence(&buffers.y_smooth, &buffers.y_prev, tol) {
                    break;
                }
            }

            // Update robustness weights for next iteration (skip last)
            if iter < niter {
                Self::update_robustness_weights(
                    y,
                    &buffers.y_smooth,
                    &mut buffers.residuals,
                    &mut buffers.robustness_weights,
                    robustness_updater,
                );
            }
        }

        // Compute standard errors if requested
        let std_errors = interval_method.map(|im| {
            Self::compute_std_errors(
                x,
                y,
                &buffers.y_smooth,
                window_size,
                &buffers.robustness_weights,
                weight_function,
                im,
            )
        });

        (
            buffers.y_smooth,
            std_errors,
            iterations_performed,
            buffers.robustness_weights,
        )
    }

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /// Smooth data using a `LowessConfig` payload.
    pub fn run_with_config(x: &[T], y: &[T], config: LowessConfig<T>) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let default_frac = T::from(0.67).unwrap_or(T::from(0.5).unwrap());
        let fraction = config.fraction.unwrap_or(default_frac);

        let executor = LowessExecutor::new()
            .fraction(fraction)
            .iterations(config.iterations)
            .delta(config.delta)
            .weight_function(config.weight_function)
            .zero_weight_fallback(config.zero_weight_fallback)
            .robustness_method(config.robustness_method)
            .boundary_policy(config.boundary_policy)
            .custom_smooth_pass(config.custom_smooth_pass);

        // Handle cross-validation if configured
        if let Some(cv_fracs) = config.cv_fractions {
            if cv_fracs.is_empty() {
                // Fallback to standard run with default fraction
                return executor.run(x, y, Some(fraction), None, None, None);
            }

            let cv_method = config.cv_method.unwrap_or(CVMethod::KFold(5));

            // Run CV to find best fraction
            let (best_frac, scores) = cv_method.run(x, y, &cv_fracs, |tx, ty, f| {
                executor.run(tx, ty, Some(f), None, None, None).smoothed
            });

            // Run final pass with best fraction
            let mut output = executor.run(
                x,
                y,
                Some(best_frac),
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            );
            output.cv_scores = Some(scores);
            output.used_fraction = best_frac;
            output
        } else {
            // Direct run (no CV)
            executor.run(
                x,
                y,
                config.fraction,
                Some(config.iterations),
                config.auto_convergence,
                config.return_variance.as_ref(),
            )
        }
    }

    /// Execute smoothing with explicit overrides for specific parameters.
    ///
    /// # Special Cases
    ///
    /// * **Insufficient data** (n < 2): Returns original y-values.
    /// * **Global regression** (fraction >= 1.0): Performs OLS on the entire dataset.
    fn run(
        &self,
        x: &[T],
        y: &[T],
        fraction: Option<T>,
        max_iter: Option<usize>,
        tolerance: Option<T>,
        confidence_method: Option<&IntervalMethod<T>>,
    ) -> ExecutorOutput<T>
    where
        T: Float + Debug + Send + Sync + 'static,
    {
        let n = x.len();
        let eff_fraction = fraction.unwrap_or(self.fraction);

        // Edge case: too few points
        if n < 2 {
            return ExecutorOutput {
                smoothed: y.to_vec(),
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
                robustness_weights: vec![T::one(); n],
            };
        }

        // Handle global regression (fraction >= 1.0)
        if eff_fraction >= T::one() {
            let smoothed = GLSModel::global_ols(x, y);
            return ExecutorOutput {
                smoothed,
                std_errors: if confidence_method.is_some() {
                    Some(vec![T::zero(); n])
                } else {
                    None
                },
                iterations: None,
                used_fraction: eff_fraction,
                cv_scores: None,
                robustness_weights: vec![T::one(); n],
            };
        }

        // Calculate window size and prepare fitter
        let window_size = Window::calculate_span(n, eff_fraction);
        let fitter = LinearRegression;
        let target_iterations = max_iter.unwrap_or(self.iterations);

        // Handle boundary padding
        let (x_in, y_in, pad_len) = if self.boundary_policy != BoundaryPolicy::Extend
            || self.boundary_policy == BoundaryPolicy::Extend
        {
            let (px, py) = apply_boundary_policy(x, y, window_size, self.boundary_policy);
            let pad = (px.len() - x.len()) / 2;
            (px, py, pad)
        } else {
            (x.to_vec(), y.to_vec(), 0)
        };

        let x_ref = &x_in;
        let y_ref = &y_in;

        // Run the iteration loop
        let (mut smoothed, mut std_errors, iterations, mut robustness_weights) = self
            .iteration_loop_with_callback(
                x_ref,
                y_ref,
                window_size,
                target_iterations,
                self.delta,
                self.weight_function,
                self.zero_weight_fallback,
                &fitter,
                &self.robustness_method,
                confidence_method,
                tolerance,
                self.custom_smooth_pass,
            );

        // Slice back to original range if padded
        if pad_len > 0 {
            smoothed.drain(0..pad_len);
            smoothed.truncate(n);

            if let Some(se) = std_errors.as_mut() {
                se.drain(0..pad_len);
                se.truncate(n);
            }

            robustness_weights.drain(0..pad_len);
            robustness_weights.truncate(n);
        }

        ExecutorOutput {
            smoothed,
            std_errors,
            iterations: if tolerance.is_some() {
                Some(iterations)
            } else {
                None
            },
            used_fraction: eff_fraction,
            cv_scores: None,
            robustness_weights,
        }
    }
}
