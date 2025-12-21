//! Interpolation and delta optimization for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides utilities for optimizing LOWESS performance through
//! delta-based point skipping and linear interpolation. When data points are
//! densely sampled, fitting every point is computationally expensive and
//! often unnecessary. This module implements the "delta optimization" that
//! fits anchor points and interpolates between them.
//!
//! ## Design notes
//!
//! * Delta controls the distance threshold for point skipping during fitting.
//! * Points closer than delta may be interpolated rather than explicitly fitted.
//! * Uses linear interpolation to fill gaps between fitted anchor points.
//! * Handles tied x-values (duplicates) by copying fitted values.
//! * Delta defaults to 1% of the x-range in `calculate_delta`.
//!   (Note: specific adapters like Streaming may default to 0.0 unless configured).
//! * All operations are generic over `Float` types to support f32 and f64.
//!
//! ## Key concepts
//!
//! ### Delta Optimization
//! Instead of fitting every point, the algorithm:
//! 1. Selects "anchor" points spaced at least delta apart
//! 2. Explicitly fits these anchor points
//! 3. Linearly interpolates between anchors for intermediate points
//!
//! This provides significant speedup on dense data with minimal accuracy loss.
//!
//! ### Linear Interpolation
//! For points between two fitted anchors at (x₀, y₀) and (x₁, y₁), the
//! interpolated value at x is:
//! ```text
//! y = y_0 + alpha * (y_1 - y_0)  where  alpha = (x - x_0) / (x_1 - x_0)
//! ```
//!
//! ### Tied Values
//! When multiple points have identical x-values, they all receive the same
//! fitted value (the value computed for that x-coordinate).
//!
//! ## Invariants
//!
//! * Input x-values must be sorted in ascending order.
//! * Delta must be non-negative and finite.
//! * Interpolation preserves monotonicity between anchor points.
//! * At least one point is always fitted (no infinite loops).
//!
//! ## Non-goals
//!
//! * This module does not perform the actual smoothing/fitting.
//! * This module does not sort the input data.
//! * This module does not provide higher-order interpolation (only linear).
//! * This module does not validate that x-values are sorted.
//!
//! ## Visibility
//!
//! This module is an internal implementation detail used by the LOWESS
//! engine. It is not part of the public API and may change without notice.

#[cfg(not(feature = "std"))]
extern crate alloc;

use crate::primitives::errors::LowessError;
use core::result::Result;
use num_traits::Float;

// ============================================================================
// Delta Calculation
// ============================================================================

/// Calculate delta parameter for interpolation optimization.
///
/// # Default behavior
///
/// If delta is `None`, computes a conservative default as 1% of the x-range:
/// ```text
/// delta = 0.01 × (max(x) - min(x))
/// ```
pub fn calculate_delta<T: Float>(delta: Option<T>, x_sorted: &[T]) -> Result<T, LowessError> {
    match delta {
        Some(d) => {
            // Validate provided delta
            if !d.is_finite() || d < T::zero() {
                return Err(LowessError::InvalidDelta(d.to_f64().unwrap_or(f64::NAN)));
            }
            Ok(d)
        }
        None => {
            // Compute default delta as 1% of x-range
            if x_sorted.is_empty() {
                Ok(T::zero())
            } else {
                let range = x_sorted[x_sorted.len() - 1] - x_sorted[0];
                Ok(T::from(0.01).unwrap() * range)
            }
        }
    }
}

// ============================================================================
// Linear Interpolation
// ============================================================================

/// Interpolate gap between two fitted anchor points.
///
/// # Special cases
///
/// * **No gap**: If current <= last_fitted + 1, no interpolation is needed
/// * **Tied x-values**: If x₁ = x₀, uses simple average of y-values
/// * **Decreasing x**: Treated same as tied values (uses average)
pub fn interpolate_gap<T: Float>(x: &[T], y_smooth: &mut [T], last_fitted: usize, current: usize) {
    // No gap to interpolate
    if current <= last_fitted + 1 {
        return;
    }

    let x0 = x[last_fitted];
    let x1 = x[current];
    let y0 = y_smooth[last_fitted];
    let y1 = y_smooth[current];

    let denom = x1 - x0;

    if denom <= T::zero() {
        // Duplicate or decreasing x-values: use simple average
        let avg = (y0 + y1) / T::from(2.0).unwrap();
        y_smooth
            .iter_mut()
            .take(current)
            .skip(last_fitted + 1)
            .for_each(|ys| *ys = avg);
        return;
    }

    // Linear interpolation
    for k in (last_fitted + 1)..current {
        let xi = x[k];
        let alpha = (xi - x0) / denom;
        y_smooth[k] = y0 + alpha * (y1 - y0);
    }
}
