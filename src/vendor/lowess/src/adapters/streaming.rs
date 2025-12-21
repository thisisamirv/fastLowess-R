//! Streaming adapter for large-scale LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the streaming execution adapter for LOWESS smoothing
//! on datasets too large to fit in memory. It divides the data into overlapping
//! chunks, processes each chunk independently, and merges the results while
//! handling boundary effects. This enables LOWESS smoothing on arbitrarily
//! large datasets with controlled memory usage.
//!
//! ## Design notes
//!
//! * Processes data in fixed-size chunks with configurable overlap.
//! * Automatically sorts data within each chunk.
//! * Merges overlapping regions using configurable strategies.
//! * Handles boundary effects at chunk edges.
//! * Supports smoothing, residuals, robustness weights, and diagnostics.
//! * Generic over `Float` types to support f32 and f64.
//! * Stateful: maintains overlap buffer between chunks.
//!
//! ## Key concepts
//!
//! ### Chunked Processing
//! The streaming adapter divides data into overlapping chunks:
//! ```text
//! Chunk 1: [==========]
//! Chunk 2:       [==========]
//! Chunk 3:             [==========]
//!          ↑overlap↑
//! ```
//!
//! ### Overlap Strategy
//! Overlap ensures smooth transitions between chunks:
//! * **Rule of thumb**: overlap = 2 × window_size
//! * Larger overlap = better boundary handling, more computation
//! * Smaller overlap = faster processing, potential edge artifacts
//!
//! ### Merge Strategies
//! When chunks overlap, values are merged using:
//! * **Average**: Simple average of overlapping values
//! * **Weighted**: Distance-weighted average
//! * **KeepFirst**: Use value from first chunk
//! * **KeepLast**: Use value from last chunk
//!
//! ### Boundary Policies
//! At dataset boundaries:
//! * **Extend**: Extend the first/last values to fill neighborhoods
//! * **Reflect**: Reflect values around the boundary
//! * **Zero**: Use zero padding at boundaries
//!
//! ### Processing Flow
//! For each chunk:
//! 1. Validate chunk data
//! 2. Sort chunk by x-values
//! 3. Perform LOWESS smoothing
//! 4. Extract non-overlapping portion
//! 5. Merge overlap with previous chunk
//! 6. Buffer overlap for next chunk
//!
//! ## Supported features
//!
//! * **Robustness iterations**: Downweight outliers iteratively
//! * **Residuals**: Differences between original and smoothed values
//! * **Robustness weights**: Return weights used for each point
//! * **Diagnostics**: RMSE, MAE, R^2, Residual SD (cumulative)
//! * **Delta optimization**: Point skipping for dense data
//! * **Configurable chunking**: Chunk size and overlap
//! * **Merge strategies**: Multiple overlap merging options
//!
//! ## Invariants
//!
//! * Chunk size must be larger than overlap.
//! * Overlap must be large enough for local smoothing.
//! * All values must be finite (no NaN or infinity).
//! * At least 2 points required per chunk.
//! * Output order matches input order within each chunk.
//!
//! ## Non-goals
//!
//! * This adapter does not support confidence/prediction intervals.
//! * This adapter does not support cross-validation.
//! * This adapter does not handle batch processing (use batch adapter).
//! * This adapter does not handle incremental updates (use online adapter).
//! * This adapter requires chunks to be provided in stream order.
//!
//! ## Visibility
//!
//! The streaming adapter is part of the public API through the high-level
//! `Lowess` builder. Direct usage of `StreamingLowess` is possible but not
//! the primary interface.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::vec::Vec;

use crate::algorithms::regression::ZeroWeightFallback;
use crate::algorithms::robustness::RobustnessMethod;
use crate::engine::executor::{LowessConfig, LowessExecutor, SmoothPassFn};
use crate::engine::output::LowessResult;
use crate::engine::validator::Validator;
use crate::evaluation::diagnostics::DiagnosticsState;
use crate::math::kernel::WeightFunction;
use crate::primitives::errors::LowessError;
use crate::primitives::partition::{BoundaryPolicy, MergeStrategy};
use crate::primitives::sorting::sort_by_x;

use core::fmt::Debug;
use core::result::Result;
use num_traits::Float;

// ============================================================================
// Streaming LOWESS Builder
// ============================================================================

/// Builder for streaming LOWESS processor.
#[derive(Debug, Clone)]
pub struct StreamingLowessBuilder<T: Float> {
    /// Chunk size for processing
    pub chunk_size: usize,

    /// Overlap between chunks
    pub overlap: usize,

    /// Smoothing fraction (span)
    pub fraction: T,

    /// Number of robustness iterations
    pub iterations: usize,

    /// Convergence tolerance for early stopping (None = disabled)
    pub auto_convergence: Option<T>,

    /// Delta parameter for interpolation
    pub delta: T,

    /// Kernel weight function
    pub weight_function: WeightFunction,

    /// Boundary handling policy
    pub boundary_policy: BoundaryPolicy,

    /// Robustness method
    pub robustness_method: RobustnessMethod,

    /// Policy for handling zero-weight neighborhoods
    pub zero_weight_fallback: ZeroWeightFallback,

    /// Merging strategy for overlapping chunks
    pub merge_strategy: MergeStrategy,

    /// Whether to return residuals
    pub compute_residuals: bool,

    /// Whether to return diagnostics
    pub return_diagnostics: bool,

    /// Whether to return robustness weights
    pub return_robustness_weights: bool,

    /// Deferred error from adapter conversion
    pub deferred_error: Option<LowessError>,

    /// Optional custom smoothing function (e.g., for parallel execution)
    pub custom_smooth_pass: Option<SmoothPassFn<T>>,

    /// Tracks if any parameter was set multiple times (for validation)
    pub(crate) duplicate_param: Option<&'static str>,
}

impl<T: Float> Default for StreamingLowessBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> StreamingLowessBuilder<T> {
    /// Create a new streaming LOWESS builder with default parameters.
    fn new() -> Self {
        Self {
            chunk_size: 5000,
            overlap: 500,
            fraction: T::from(0.1).unwrap(),
            iterations: 2,
            delta: T::zero(),
            weight_function: WeightFunction::default(),
            boundary_policy: BoundaryPolicy::default(),
            robustness_method: RobustnessMethod::default(),
            zero_weight_fallback: ZeroWeightFallback::default(),
            merge_strategy: MergeStrategy::default(),
            compute_residuals: false,
            return_diagnostics: false,
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

    /// Set the delta parameter for interpolation optimization.
    pub fn delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
    }

    /// Set kernel weight function.
    pub fn weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.weight_function = weight_function;
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
    // Streaming-Specific Setters
    // ========================================================================

    /// Set chunk size for processing.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set overlap between chunks.
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the merge strategy for overlapping chunks.
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Set whether to return diagnostics.
    pub fn return_diagnostics(mut self, return_diagnostics: bool) -> Self {
        self.return_diagnostics = return_diagnostics;
        self
    }

    // ========================================================================
    // Build Method
    // ========================================================================

    /// Build the streaming processor.
    pub fn build(self) -> Result<StreamingLowess<T>, LowessError> {
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
        Validator::validate_delta(self.delta)?;

        // Validate chunk size
        Validator::validate_chunk_size(self.chunk_size, 10)?;

        // Validate overlap
        Validator::validate_overlap(self.overlap, self.chunk_size)?;

        let has_diag = self.return_diagnostics;
        Ok(StreamingLowess {
            config: self,
            overlap_buffer_x: Vec::new(),
            overlap_buffer_y: Vec::new(),
            overlap_buffer_smoothed: Vec::new(),
            overlap_buffer_robustness_weights: Vec::new(),
            diagnostics_state: if has_diag {
                Some(DiagnosticsState::new())
            } else {
                None
            },
        })
    }
}

// ============================================================================
// Streaming LOWESS Processor
// ============================================================================

/// Streaming LOWESS processor for large datasets.
pub struct StreamingLowess<T: Float> {
    config: StreamingLowessBuilder<T>,
    overlap_buffer_x: Vec<T>,
    overlap_buffer_y: Vec<T>,
    overlap_buffer_smoothed: Vec<T>,
    overlap_buffer_robustness_weights: Vec<T>,
    diagnostics_state: Option<DiagnosticsState<T>>,
}

impl<T: Float + Debug + Send + Sync + 'static> StreamingLowess<T> {
    /// Process a chunk of data.
    pub fn process_chunk(&mut self, x: &[T], y: &[T]) -> Result<LowessResult<T>, LowessError> {
        // Validate inputs using standard validator
        Validator::validate_inputs(x, y)?;

        // Sort chunk by x
        let sorted = sort_by_x(x, y);

        // Configure LOWESS for this chunk
        // Combine with overlap from previous chunk
        let prev_overlap_len = self.overlap_buffer_smoothed.len();
        let (combined_x, combined_y) = if self.overlap_buffer_x.is_empty() {
            (sorted.x.clone(), sorted.y.clone())
        } else {
            let mut cx = core::mem::take(&mut self.overlap_buffer_x);
            cx.extend_from_slice(&sorted.x);
            let mut cy = core::mem::take(&mut self.overlap_buffer_y);
            cy.extend_from_slice(&sorted.y);
            (cx, cy)
        };

        let zero_flag = self.config.zero_weight_fallback.to_u8();

        let config = LowessConfig {
            fraction: Some(self.config.fraction),
            iterations: self.config.iterations,
            delta: self.config.delta,
            weight_function: self.config.weight_function,
            zero_weight_fallback: zero_flag,
            robustness_method: self.config.robustness_method,
            boundary_policy: self.config.boundary_policy,
            cv_fractions: None,
            cv_method: None,
            auto_convergence: self.config.auto_convergence,
            return_variance: None,
            custom_smooth_pass: self.config.custom_smooth_pass,
        };
        // Execute LOWESS on combined data
        let result = LowessExecutor::run_with_config(&combined_x, &combined_y, config);
        let smoothed = result.smoothed;

        // Determine how much to return vs buffer
        let combined_len = combined_x.len();
        let overlap_start = combined_len.saturating_sub(self.config.overlap);
        let return_start = prev_overlap_len;

        // Build output: merged overlap (if any) + new data
        let mut y_smooth_out = Vec::new();
        if prev_overlap_len > 0 {
            // Merge the overlap region
            let prev_smooth = core::mem::take(&mut self.overlap_buffer_smoothed);
            for (i, (&prev_val, &curr_val)) in prev_smooth
                .iter()
                .zip(smoothed.iter())
                .take(prev_overlap_len)
                .enumerate()
            {
                let merged = match self.config.merge_strategy {
                    MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                    MergeStrategy::WeightedAverage => {
                        let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                        prev_val * (T::one() - weight) + curr_val * weight
                    }
                    MergeStrategy::TakeFirst => prev_val,
                    MergeStrategy::TakeLast => curr_val,
                };
                y_smooth_out.push(merged);
            }
        }

        // Merge robustness weights if requested
        let mut rob_weights_out = if self.config.return_robustness_weights {
            Some(Vec::new())
        } else {
            None
        };

        if let Some(ref mut rw_out) = rob_weights_out {
            if prev_overlap_len > 0 {
                let prev_rw = core::mem::take(&mut self.overlap_buffer_robustness_weights);
                for (i, (&prev_val, &curr_val)) in prev_rw
                    .iter()
                    .zip(result.robustness_weights.iter())
                    .take(prev_overlap_len)
                    .enumerate()
                {
                    let merged = match self.config.merge_strategy {
                        MergeStrategy::Average => (prev_val + curr_val) / T::from(2.0).unwrap(),
                        MergeStrategy::WeightedAverage => {
                            let weight = T::from(i as f64 / prev_overlap_len as f64).unwrap();
                            prev_val * (T::one() - weight) + curr_val * weight
                        }
                        MergeStrategy::TakeFirst => prev_val,
                        MergeStrategy::TakeLast => curr_val,
                    };
                    rw_out.push(merged);
                }
            }
        }

        // Add non-overlap portion
        if return_start < overlap_start {
            y_smooth_out.extend_from_slice(&smoothed[return_start..overlap_start]);
            if let Some(ref mut rw_out) = rob_weights_out {
                rw_out.extend_from_slice(&result.robustness_weights[return_start..overlap_start]);
            }
        }

        // Calculate residuals for output
        let residuals_out = if self.config.compute_residuals {
            let y_slice = &combined_y[return_start..return_start + y_smooth_out.len()];
            Some(
                y_slice
                    .iter()
                    .zip(y_smooth_out.iter())
                    .map(|(y, s)| *y - *s)
                    .collect(),
            )
        } else {
            None
        };

        // Buffer overlap for next chunk
        if overlap_start < combined_len {
            self.overlap_buffer_x = combined_x[overlap_start..].to_vec();
            self.overlap_buffer_y = combined_y[overlap_start..].to_vec();
            self.overlap_buffer_smoothed = smoothed[overlap_start..].to_vec();
            if self.config.return_robustness_weights {
                self.overlap_buffer_robustness_weights =
                    result.robustness_weights[overlap_start..].to_vec();
            }
        } else {
            self.overlap_buffer_x.clear();
            self.overlap_buffer_y.clear();
            self.overlap_buffer_smoothed.clear();
            self.overlap_buffer_robustness_weights.clear();
        }

        // Note: We return results in sorted order (by x) for streaming chunks.
        // Unsorting partial results is ambiguous since we only return a subset of the chunk.
        // The full batch adapter handles global unsorting when processing complete datasets.
        let x_out = combined_x[return_start..return_start + y_smooth_out.len()].to_vec();

        // Update diagnostics cumulatively
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            let y_emitted = &combined_y[return_start..return_start + y_smooth_out.len()];
            state.update(y_emitted, &y_smooth_out);
            Some(state.finalize())
        } else {
            None
        };

        Ok(LowessResult {
            x: x_out,
            y: y_smooth_out,
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals: residuals_out,
            robustness_weights: rob_weights_out,
            diagnostics,
            iterations_used: result.iterations,
            fraction_used: self.config.fraction,
            cv_scores: None,
        })
    }

    /// Finalize processing and get any remaining buffered data.
    pub fn finalize(&mut self) -> Result<LowessResult<T>, LowessError> {
        if self.overlap_buffer_x.is_empty() {
            return Ok(LowessResult {
                x: Vec::new(),
                y: Vec::new(),
                standard_errors: None,
                confidence_lower: None,
                confidence_upper: None,
                prediction_lower: None,
                prediction_upper: None,
                residuals: None,
                robustness_weights: None,
                diagnostics: None,
                iterations_used: None,
                fraction_used: self.config.fraction,
                cv_scores: None,
            });
        }

        // Return buffered overlap data
        let residuals = if self.config.compute_residuals {
            let mut res = Vec::with_capacity(self.overlap_buffer_x.len());
            for (i, &smoothed) in self.overlap_buffer_smoothed.iter().enumerate() {
                res.push(self.overlap_buffer_y[i] - smoothed);
            }
            Some(res)
        } else {
            None
        };

        let robustness_weights = if self.config.return_robustness_weights {
            Some(core::mem::take(&mut self.overlap_buffer_robustness_weights))
        } else {
            None
        };

        // Update diagnostics for the final overlap
        let diagnostics = if let Some(ref mut state) = self.diagnostics_state {
            state.update(&self.overlap_buffer_y, &self.overlap_buffer_smoothed);
            Some(state.finalize())
        } else {
            None
        };

        let result = LowessResult {
            x: self.overlap_buffer_x.clone(),
            y: self.overlap_buffer_smoothed.clone(),
            standard_errors: None,
            confidence_lower: None,
            confidence_upper: None,
            prediction_lower: None,
            prediction_upper: None,
            residuals,
            robustness_weights,
            diagnostics,
            iterations_used: None,
            fraction_used: self.config.fraction,
            cv_scores: None,
        };

        // Clear buffers
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();
        self.overlap_buffer_robustness_weights.clear();

        Ok(result)
    }

    /// Reset the processor state.
    ///
    /// Clears all buffered overlap data. Useful when starting a new stream
    /// or handling gaps in the data.
    pub fn reset(&mut self) {
        self.overlap_buffer_x.clear();
        self.overlap_buffer_y.clear();
        self.overlap_buffer_smoothed.clear();
        self.overlap_buffer_robustness_weights.clear();
    }
}
