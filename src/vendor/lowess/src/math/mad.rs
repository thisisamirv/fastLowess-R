//! Median Absolute Deviation (MAD) computation for robust scale estimation.
//!
//! ## Purpose
//!
//! This module provides robust scale estimation using the Median Absolute
//! Deviation (MAD), which is resistant to outliers. MAD is used in LOWESS
//! for computing robustness weights that downweight outliers in iterative
//! refinement.
//!
//! ## Design notes
//!
//! * MAD is computed as: MAD = median(|rᵢ - median(r)|)
//! * Uses in-place selection algorithms for O(n) average-case performance.
//! * Reuses allocated buffers to minimize memory allocations.
//! * All functions are generic over `Float` types to support f32 and f64.
//! * Supports both `std` and `no_std` environments.
//!
//! ## Key concepts
//!
//! ### Median Absolute Deviation
//!
//! MAD is a robust measure of variability:
//! MAD = median(|r_i - median(r)|)
//! It is significantly more resistant to outliers than the standard deviation
//! and is the standard scale estimator in robust statistics.
//!
//! ### Robustness
//!
//! MAD has a breakdown point of 50%, meaning it remains a reliable scale
//! estimate even when up to half of the observations are extreme outliers.
//! This makes it ideal for iterative weight refinement in LOWESS.
//!
//! ### Computational Efficiency
//!
//! ### Computational Efficiency
//!
//! Computed in O(n) average-case time using Quickselect (`select_nth_unstable`).
//! This avoids the O(n log n) overhead of full sorting.
//!
//! ## Invariants
//!
//! * MAD >= 0 for any input.
//! * MAD = 0 if |residuals| <= 1 or if all values are identical.
//! * Handles even and odd population sizes with standard median averaging.
//!
//! ## Non-goals
//!
//! * This module does not provide weighted MAD variants.
//! * This module does not handle non-finite values (NaN/Inf).
//! * This module is an internal utility and not part of the stable public API.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// MAD Computation
// ============================================================================

/// Compute the Median Absolute Deviation (MAD) of a slice of residuals.
///
/// # Formula
///
/// Calculated as:
/// ```text
/// MAD = median(|r_i - median(r)|)
/// ```
pub fn compute_mad<T: Float>(residuals: &[T]) -> T {
    let n = residuals.len();

    // Edge case: need at least 2 points for meaningful MAD
    if n <= 1 {
        return T::zero();
    }

    // Helper function to compute median in-place
    let median_inplace = |vals: &mut [T]| -> T {
        let n = vals.len();

        if n == 0 {
            return T::zero();
        }
        if n == 1 {
            return vals[0];
        }

        let mid = n / 2;

        if n % 2 == 0 {
            // Even length: average of two middle values
            vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let upper = vals[mid];

            // Find the largest value in the lower half
            let lower = vals[..mid].iter().copied().fold(T::neg_infinity(), T::max);

            (lower + upper) / T::from(2.0).unwrap()
        } else {
            // Odd length: middle value
            vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            vals[mid]
        }
    };

    // Step 1: Compute median of residuals
    let mut vals: Vec<T> = residuals.to_vec();
    let median: T = median_inplace(&mut vals);

    // Step 2: Compute absolute deviations from median
    // Reuse vals buffer to avoid second allocation
    for val in vals.iter_mut() {
        *val = (*val - median).abs();
    }

    // Step 3: Return median of absolute deviations
    median_inplace(&mut vals)
}
