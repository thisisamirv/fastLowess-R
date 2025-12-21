//! High-level API for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the primary user-facing entry point for LOWESS. It
//! implements a fluent builder pattern for configuring regression parameters
//! and choosing an execution adapter (Batch, Streaming, or Online).
//!
//! ## Design notes
//!
//! * **Ergonomic**: Fluent builder with sensible defaults for all parameters.
//! * **Polymorphic**: Uses marker types to transition to specialized adapter builders.
//! * **Validated**: Core parameters are validated during adapter construction.
//! * **Type-Safe**: Generic over `Float` types for flexible precision.
//!
//! ## Key concepts
//!
//! ### Execution Adapters
//!
//! * **Batch**: Standard in-memory processing for complete datasets.
//! * **Streaming**: Chunked processing with overlap merging for large datasets.
//! * **Online**: Incremental updates via a sliding window for real-time data.
//!
//! ### Configuration Flow
//!
//! 1. Create a [`LowessBuilder`] via `Lowess::new()`.
//! 2. Chain configuration methods (`.fraction()`, `.iterations()`, etc.).
//! 3. Select an adapter via `.adapter(Adapter::Batch)` to get an execution builder.
//!
//! ## Visibility
//!
//! This is the primary public API. Types re-exported here are considered stable.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use core::result;
use num_traits::Float;

// Internal adapters
use crate::adapters::batch::BatchLowessBuilder;
use crate::adapters::online::OnlineLowessBuilder;
use crate::adapters::streaming::StreamingLowessBuilder;
pub use crate::engine::executor::SmoothPassFn;

// Publicly re-exported types
pub use crate::algorithms::regression::ZeroWeightFallback;
pub use crate::algorithms::robustness::RobustnessMethod;
pub use crate::engine::output::LowessResult;
pub use crate::evaluation::cv::CVMethod;
pub use crate::evaluation::intervals::IntervalMethod;
pub use crate::math::kernel::WeightFunction;
pub use crate::primitives::errors::LowessError;
pub use crate::primitives::partition::{BoundaryPolicy, MergeStrategy, UpdateMode};

/// Result type alias for LOWESS operations.
pub type Result<T> = result::Result<T, LowessError>;

/// Marker types for selecting execution adapters.
#[allow(non_snake_case)]
pub mod Adapter {
    pub use super::{Batch, Online, Streaming};
}

/// Fluent builder for configuring LOWESS parameters and execution modes.
#[derive(Debug, Clone)]
pub struct LowessBuilder<T> {
    /// Smoothing fraction (0..1].
    pub fraction: Option<T>,

    /// Robustness iterations.
    pub iterations: Option<usize>,

    /// Threshold for skipping fitting (delta-optimization).
    pub delta: Option<T>,

    /// Kernel weight function.
    pub weight_function: Option<WeightFunction>,

    /// Outlier downweighting method.
    pub robustness_method: Option<RobustnessMethod>,

    /// interval estimation configuration.
    pub interval_type: Option<IntervalMethod<T>>,

    /// Candidate bandwidths for cross-validation.
    pub cv_fractions: Option<Vec<T>>,

    /// CV strategy (K-Fold/LOOCV).
    pub cv_method: Option<CVMethod>,

    /// Relative convergence tolerance.
    pub auto_convergence: Option<T>,

    /// Enable performance/statistical diagnostics.
    pub return_diagnostics: bool,

    /// Return original residuals r_i.
    pub compute_residuals: bool,

    /// Return final robustness weights w_i.
    pub return_robustness_weights: bool,

    /// Policy for handling data boundaries (default: Extend).
    pub boundary_policy: Option<BoundaryPolicy>,

    /// Behavior when local neighborhood weights are zero (default: UseLocalMean).
    pub zero_weight_fallback: Option<ZeroWeightFallback>,

    /// Merging strategy for overlapping chunks (Streaming only).
    pub merge_strategy: Option<MergeStrategy>,

    /// Incremental update mode (Online only).
    pub update_mode: Option<UpdateMode>,

    /// Chunk size for streaming (Streaming only).
    pub chunk_size: Option<usize>,

    /// Overlap size for streaming chunks (Streaming only).
    pub overlap: Option<usize>,

    /// Window capacity for sliding window (Online only).
    pub window_capacity: Option<usize>,

    /// Minimum points required for a valid fit (Online only).
    pub min_points: Option<usize>,

    /// Custom smooth pass function (e.g., for parallel execution).
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Tracks if any parameter was set multiple times (for validation).
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for LowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> LowessBuilder<T> {
    /// Select an execution adapter to transition to an execution builder.
    pub fn adapter<A>(self, _adapter: A) -> A::Output
    where
        A: LowessAdapter<T>,
    {
        A::convert(self)
    }

    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            fraction: None,
            iterations: None,
            delta: None,
            weight_function: None,
            robustness_method: None,
            interval_type: None,
            cv_fractions: None,
            cv_method: None,
            auto_convergence: None,
            return_diagnostics: false,
            compute_residuals: false,
            return_robustness_weights: false,
            boundary_policy: None,
            zero_weight_fallback: None,
            merge_strategy: None,
            update_mode: None,
            chunk_size: None,
            overlap: None,
            window_capacity: None,
            min_points: None,
            custom_smooth_pass: None,
            duplicate_param: None,
        }
    }

    /// Set behavior for handling zero-weight neighborhoods.
    pub fn zero_weight_fallback(mut self, policy: ZeroWeightFallback) -> Self {
        if self.zero_weight_fallback.is_some() {
            self.duplicate_param = Some("zero_weight_fallback");
        }
        self.zero_weight_fallback = Some(policy);
        self
    }

    /// Set the boundary handling policy.
    pub fn boundary_policy(mut self, policy: BoundaryPolicy) -> Self {
        if self.boundary_policy.is_some() {
            self.duplicate_param = Some("boundary_policy");
        }
        self.boundary_policy = Some(policy);
        self
    }

    /// Set the merging strategy for overlapping chunks (Streaming only).
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        if self.merge_strategy.is_some() {
            self.duplicate_param = Some("merge_strategy");
        }
        self.merge_strategy = Some(strategy);
        self
    }

    /// Set the incremental update mode (Online only).
    pub fn update_mode(mut self, mode: UpdateMode) -> Self {
        if self.update_mode.is_some() {
            self.duplicate_param = Some("update_mode");
        }
        self.update_mode = Some(mode);
        self
    }

    /// Set the chunk size for streaming (Streaming only).
    pub fn chunk_size(mut self, size: usize) -> Self {
        if self.chunk_size.is_some() {
            self.duplicate_param = Some("chunk_size");
        }
        self.chunk_size = Some(size);
        self
    }

    /// Set the overlap size for streaming chunks (Streaming only).
    pub fn overlap(mut self, overlap: usize) -> Self {
        if self.overlap.is_some() {
            self.duplicate_param = Some("overlap");
        }
        self.overlap = Some(overlap);
        self
    }

    /// Set the window capacity for online processing (Online only).
    pub fn window_capacity(mut self, capacity: usize) -> Self {
        if self.window_capacity.is_some() {
            self.duplicate_param = Some("window_capacity");
        }
        self.window_capacity = Some(capacity);
        self
    }

    /// Set the minimum points required for a valid fit (Online only).
    pub fn min_points(mut self, points: usize) -> Self {
        if self.min_points.is_some() {
            self.duplicate_param = Some("min_points");
        }
        self.min_points = Some(points);
        self
    }

    /// Set a custom smooth pass function for execution.
    pub fn custom_smooth_pass(mut self, pass: SmoothPassFn<T>) -> Self {
        self.custom_smooth_pass = Some(pass);
        self
    }

    /// Set the smoothing fraction (bandwidth alpha).
    pub fn fraction(mut self, fraction: T) -> Self {
        if self.fraction.is_some() {
            self.duplicate_param = Some("fraction");
        }
        self.fraction = Some(fraction);
        self
    }

    /// Set the number of robustness iterations (typically 0-4).
    pub fn iterations(mut self, iterations: usize) -> Self {
        if self.iterations.is_some() {
            self.duplicate_param = Some("iterations");
        }
        self.iterations = Some(iterations);
        self
    }

    /// Set the delta parameter for interpolation-based optimization.
    pub fn delta(mut self, delta: T) -> Self {
        if self.delta.is_some() {
            self.duplicate_param = Some("delta");
        }
        self.delta = Some(delta);
        self
    }

    /// Set the kernel weight function.
    pub fn weight_function(mut self, wf: WeightFunction) -> Self {
        if self.weight_function.is_some() {
            self.duplicate_param = Some("weight_function");
        }
        self.weight_function = Some(wf);
        self
    }

    /// Set the robustness weighting method.
    pub fn robustness_method(mut self, rm: RobustnessMethod) -> Self {
        if self.robustness_method.is_some() {
            self.duplicate_param = Some("robustness_method");
        }
        self.robustness_method = Some(rm);
        self
    }

    /// Enable standard error computation.
    pub fn return_se(mut self) -> Self {
        if self.interval_type.is_none() {
            self.interval_type = Some(IntervalMethod::se());
        }
        self
    }

    /// Enable confidence intervals at the specified level (e.g., 0.95).
    pub fn confidence_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.confidence) {
            self.duplicate_param = Some("confidence_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.prediction => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::confidence(level),
        });
        self
    }

    /// Enable prediction intervals at the specified level.
    pub fn prediction_intervals(mut self, level: T) -> Self {
        if self.interval_type.as_ref().is_some_and(|it| it.prediction) {
            self.duplicate_param = Some("prediction_intervals");
        }
        self.interval_type = Some(match self.interval_type {
            Some(existing) if existing.confidence => IntervalMethod {
                level,
                confidence: true,
                prediction: true,
                se: true,
            },
            _ => IntervalMethod::prediction(level),
        });
        self
    }

    /// Enable automatic bandwidth selection via cross-validation.
    pub fn cross_validate(
        mut self,
        fractions: &[T],
        method: CrossValidationStrategy,
        k: Option<usize>,
    ) -> Self {
        if self.cv_fractions.is_some() {
            self.duplicate_param = Some("cross_validate");
        }
        self.cv_fractions = Some(fractions.to_vec());
        self.cv_method = Some(match method {
            CrossValidationStrategy::KFold => CVMethod::KFold(k.unwrap_or(5)),
            CrossValidationStrategy::LeaveOneOut | CrossValidationStrategy::LOOCV => {
                CVMethod::LOOCV
            }
        });
        self
    }

    /// Enable automatic convergence detection based on relative change.
    pub fn auto_converge(mut self, tolerance: T) -> Self {
        if self.auto_convergence.is_some() {
            self.duplicate_param = Some("auto_converge");
        }
        self.auto_convergence = Some(tolerance);
        self
    }

    /// Include statistical diagnostics (Metric, R², etc.) in output.
    pub fn return_diagnostics(mut self) -> Self {
        self.return_diagnostics = true;
        self
    }

    /// Include residuals in output.
    pub fn return_residuals(mut self) -> Self {
        self.compute_residuals = true;
        self
    }

    /// Include final robustness weights in output.
    pub fn return_robustness_weights(mut self) -> Self {
        self.return_robustness_weights = true;
        self
    }
}

/// Trait for transitioning from a generic builder to an execution builder.
pub trait LowessAdapter<T: Float> {
    /// The output execution builder.
    type Output;

    /// Convert a generic [`LowessBuilder`] into a specialized execution builder.
    fn convert(builder: LowessBuilder<T>) -> Self::Output;
}

/// Marker for in-memory batch processing.
#[derive(Debug, Clone, Copy)]
pub struct Batch;

impl<T: Float> LowessAdapter<T> for Batch {
    type Output = BatchLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        BatchLowessBuilder {
            fraction: builder.fraction.unwrap_or_else(|| T::from(0.67).unwrap()),
            iterations: builder.iterations.unwrap_or(3),
            delta: builder.delta,
            weight_function: builder.weight_function.unwrap_or_default(),
            robustness_method: builder.robustness_method.unwrap_or_default(),
            interval_type: builder.interval_type,
            cv_fractions: builder.cv_fractions,
            cv_method: builder.cv_method,
            auto_convergence: builder.auto_convergence,
            return_diagnostics: builder.return_diagnostics,
            compute_residuals: builder.compute_residuals,
            return_robustness_weights: builder.return_robustness_weights,
            zero_weight_fallback: builder.zero_weight_fallback.unwrap_or_default(),
            boundary_policy: builder.boundary_policy.unwrap_or_default(),
            deferred_error: None,
            custom_smooth_pass: builder.custom_smooth_pass,
            duplicate_param: builder.duplicate_param,
        }
    }
}

/// Marker for chunked streaming processing.
#[derive(Debug, Clone, Copy)]
pub struct Streaming;

impl<T: Float> LowessAdapter<T> for Streaming {
    type Output = StreamingLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        let mut result = StreamingLowessBuilder::default();

        // Override with user-provided values
        if let Some(chunk_size) = builder.chunk_size {
            result.chunk_size = chunk_size;
        }
        if let Some(overlap) = builder.overlap {
            result.overlap = overlap;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(delta) = builder.delta {
            result.delta = delta;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }
        if let Some(ms) = builder.merge_strategy {
            result.merge_strategy = ms;
        }

        result.compute_residuals = builder.compute_residuals;
        result.return_diagnostics = builder.return_diagnostics;
        result.return_robustness_weights = builder.return_robustness_weights;
        result.auto_convergence = builder.auto_convergence;
        result.custom_smooth_pass = builder.custom_smooth_pass;
        result.duplicate_param = builder.duplicate_param;

        result
    }
}

/// Marker for incremental online processing.
#[derive(Debug, Clone, Copy)]
pub struct Online;

impl<T: Float> LowessAdapter<T> for Online {
    type Output = OnlineLowessBuilder<T>;

    fn convert(builder: LowessBuilder<T>) -> Self::Output {
        let mut result = OnlineLowessBuilder::default();

        // Override with user-provided values
        if let Some(window_capacity) = builder.window_capacity {
            result.window_capacity = window_capacity;
        }
        if let Some(min_points) = builder.min_points {
            result.min_points = min_points;
        }
        if let Some(fraction) = builder.fraction {
            result.fraction = fraction;
        }
        if let Some(iterations) = builder.iterations {
            result.iterations = iterations;
        }
        if let Some(delta) = builder.delta {
            result.delta = delta;
        }
        if let Some(wf) = builder.weight_function {
            result.weight_function = wf;
        }
        if let Some(um) = builder.update_mode {
            result.update_mode = um;
        }
        if let Some(rm) = builder.robustness_method {
            result.robustness_method = rm;
        }
        if let Some(bp) = builder.boundary_policy {
            result.boundary_policy = bp;
        }
        if let Some(zwf) = builder.zero_weight_fallback {
            result.zero_weight_fallback = zwf;
        }

        result.compute_residuals = builder.compute_residuals;
        result.return_robustness_weights = builder.return_robustness_weights;
        result.auto_convergence = builder.auto_convergence;
        result.custom_smooth_pass = builder.custom_smooth_pass;
        result.duplicate_param = builder.duplicate_param;

        result
    }
}

/// Strategy for dataset splitting during cross-validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation (user-defined k folds).
    KFold,

    /// Leave-one-out cross-validation (n folds).
    LeaveOneOut,

    /// Alias for `LeaveOneOut`.
    LOOCV,
}
