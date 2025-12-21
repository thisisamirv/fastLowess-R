//! Input validation for LOWESS configuration and data.
//!
//! ## Purpose
//!
//! This module provides comprehensive validation functions for LOWESS
//! configuration parameters and input data. It ensures that all inputs
//! meet the requirements for successful smoothing before any computation
//! begins, providing clear error messages when validation fails.
//!
//! ## Design notes
//!
//! * All validation is performed upfront before smoothing begins.
//! * Validation is fail-fast: returns on first error encountered.
//! * Error messages include specific values and context for debugging.
//! * Validation is generic over `Float` types to support f32 and f64.
//! * Checks are ordered from cheap to expensive for efficiency.
//! * Uses cache-friendly combined loops where possible.
//!
//! ## Validated parameters
//!
//! * **Input data**: Non-empty, matching lengths, sufficient points, all finite
//! * **Fraction**: In (0, 1] and finite
//! * **Delta**: Non-negative and finite
//! * **Interval level**: In (0, 1) and finite
//! * **CV fractions**: Non-empty, all in (0, 1], all finite
//! * **Auto-convergence tolerance**: Positive and finite
//! * **Chunk size**: Meets minimum requirements (streaming)
//! * **Overlap**: Less than chunk size (streaming)
//! * **Window capacity**: Meets minimum requirements (online)
//! * **Min points**: At least 2 and at most window capacity (online)
//!
//! ## Key concepts
//!
//! ### Fail-Fast Validation
//!
//! Validation stops at the first error encountered, returning immediately
//! with a descriptive [`LowessError`]. This avoids unnecessary allocations
//! and computation while providing quick feedback.
//!
//! ### Finite Value Checks
//!
//! All floating-point values (inputs and parameters) must be finite (not NaN
//! or infinity). This prevents numerical instability and ensures that result
//! metrics remain meaningful.
//!
//! ### Regression Requirements
//!
//! LOWESS requires at least 2 points to perform a local linear regression.
//! Additional constraints are enforced for specific use cases (e.g., minimum
//! points for online smoothing or overlap bounds for streaming).
//!
//! ### Parameter Bounds
//!
//! * **Fraction**: Must be in (0, 1] to ensure non-empty windows.
//! * **Delta**: Must be >= 0 for optimization.
//! * **Tolerance**: Must be > 0 for convergence checking.
//!
//! ## Invariants
//!
//! * All validated inputs satisfy their respective mathematical constraints.
//! * Validation logic is deterministic and side-effect free.
//! * Error messages are context-aware and include problematic values.
//!
//! ## Non-goals
//!
//! * This module does not sort, transform, or filter input data.
//! * This module does not provide automatic correction of invalid inputs.
//! * This module does not perform the smoothing or optimization itself.
//!
//! ## Visibility
//!
//! This module is an internal implementation detail used by the LOWESS
//! builder and adapters. It is not part of the public API and may change
//! without notice.

#[cfg(not(feature = "std"))]
use alloc::format;

use crate::primitives::errors::LowessError;
use num_traits::Float;

// ============================================================================
// Validator
// ============================================================================

/// Validation utility for LOWESS configuration and input data.
///
/// Provides static methods for validating various LOWESS parameters and
/// input data. All methods return `Result<(), LowessError>` and fail fast
/// upon identifying the first violation.
pub struct Validator;

impl Validator {
    // ========================================================================
    // Core Input Validation
    // ========================================================================

    /// Validate input arrays for LOWESS smoothing.
    pub fn validate_inputs<T: Float>(x: &[T], y: &[T]) -> Result<(), LowessError> {
        // Check 1: Non-empty arrays
        if x.is_empty() || y.is_empty() {
            return Err(LowessError::EmptyInput);
        }

        // Check 2: Matching lengths
        let n = x.len();
        if n != y.len() {
            return Err(LowessError::MismatchedInputs {
                x_len: n,
                y_len: y.len(),
            });
        }

        // Check 3: Sufficient points for regression
        if n < 2 {
            return Err(LowessError::TooFewPoints { got: n, min: 2 });
        }

        // Check 4: All values finite (combined loop for cache locality)
        for i in 0..n {
            if !x[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "x[{}]={}",
                    i,
                    x[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
            if !y[i].is_finite() {
                return Err(LowessError::InvalidNumericValue(format!(
                    "y[{}]={}",
                    i,
                    y[i].to_f64().unwrap_or(f64::NAN)
                )));
            }
        }

        Ok(())
    }

    // ========================================================================
    // Parameter Validation
    // ========================================================================

    /// Validate the delta optimization parameter.
    pub fn validate_delta<T: Float>(delta: T) -> Result<(), LowessError> {
        if !delta.is_finite() || delta < T::zero() {
            return Err(LowessError::InvalidDelta(
                delta.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate the smoothing fraction (bandwidth) parameter.
    pub fn validate_fraction<T: Float>(fraction: T) -> Result<(), LowessError> {
        if !fraction.is_finite() || fraction <= T::zero() || fraction > T::one() {
            return Err(LowessError::InvalidFraction(
                fraction.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate the number of robustness iterations.
    ///
    /// # Notes
    ///
    /// * 0 iterations means initial fit only (no robustness).
    /// * Maximum of 1000 iterations to prevent excessive computation.
    pub fn validate_iterations(iterations: usize) -> Result<(), LowessError> {
        const MAX_ITERATIONS: usize = 1000;
        if iterations > MAX_ITERATIONS {
            return Err(LowessError::InvalidIterations(iterations));
        }
        Ok(())
    }

    /// Validate the confidence/prediction interval level.
    pub fn validate_interval_level<T: Float>(level: T) -> Result<(), LowessError> {
        if !level.is_finite() || level <= T::zero() || level >= T::one() {
            return Err(LowessError::InvalidIntervals(
                level.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    /// Validate a collection of candidate fractions for cross-validation.
    pub fn validate_cv_fractions<T: Float>(fracs: &[T]) -> Result<(), LowessError> {
        if fracs.is_empty() {
            return Err(LowessError::InvalidFraction(0.0));
        }

        for &f in fracs {
            if !f.is_finite() || f <= T::zero() || f > T::one() {
                return Err(LowessError::InvalidFraction(f.to_f64().unwrap_or(0.0)));
            }
        }

        Ok(())
    }

    /// Validate the auto-convergence tolerance.
    pub fn validate_tolerance<T: Float>(tol: T) -> Result<(), LowessError> {
        if !tol.is_finite() || tol <= T::zero() {
            return Err(LowessError::InvalidTolerance(
                tol.to_f64().unwrap_or(f64::NAN),
            ));
        }
        Ok(())
    }

    // ========================================================================
    // Adapter-Specific Validation
    // ========================================================================

    /// Validate the chunk size for shared processing in streaming mode.
    pub fn validate_chunk_size(chunk_size: usize, min: usize) -> Result<(), LowessError> {
        if chunk_size < min {
            return Err(LowessError::InvalidChunkSize {
                got: chunk_size,
                min,
            });
        }
        Ok(())
    }

    /// Validate the overlap between consecutive chunks in streaming mode.
    pub fn validate_overlap(overlap: usize, chunk_size: usize) -> Result<(), LowessError> {
        if overlap >= chunk_size {
            return Err(LowessError::InvalidOverlap {
                overlap,
                chunk_size,
            });
        }
        Ok(())
    }

    /// Validate the maximum capacity of the sliding window in online mode.
    pub fn validate_window_capacity(window_capacity: usize, min: usize) -> Result<(), LowessError> {
        if window_capacity < min {
            return Err(LowessError::InvalidWindowCapacity {
                got: window_capacity,
                min,
            });
        }
        Ok(())
    }

    /// Validate the activation threshold for online smoothing.
    pub fn validate_min_points(
        min_points: usize,
        window_capacity: usize,
    ) -> Result<(), LowessError> {
        if min_points < 2 || min_points > window_capacity {
            return Err(LowessError::InvalidMinPoints {
                got: min_points,
                window_capacity,
            });
        }
        Ok(())
    }

    /// Validate that no parameters were set multiple times in the builder.
    pub fn validate_no_duplicates(
        duplicate_param: Option<&'static str>,
    ) -> Result<(), LowessError> {
        if let Some(param) = duplicate_param {
            return Err(LowessError::DuplicateParameter { parameter: param });
        }
        Ok(())
    }
}
