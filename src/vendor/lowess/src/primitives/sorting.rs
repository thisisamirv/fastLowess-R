//! Sorting utilities for LOWESS input data.
//!
//! ## Purpose
//!
//! This module provides utilities for sorting input data by x-coordinates and
//! mapping results back to the original order.
//!
//! ## Design notes
//!
//! * **Stability**: Uses stable sorting to preserve the relative order of equal x-values.
//! * **Robustness**: Non-finite values (NaN, Inf) are moved to the end of the sequence.
//! * **Efficiency**: Maintains an O(n) index mapping for restoring original order.
//!
//! ## Key concepts
//!
//! ### Sort-Process-Unsort Pattern
//! 1. **Sort**: Input data is sorted by x-coordinates, creating an index mapping.
//! 2. **Process**: LOWESS smoothing operates on the sorted sequence.
//! 3. **Unsort**: Results are mapped back to original indices in O(n) time.
//!
//! ## Invariants
//!
//! * Sorted x-values are strictly non-decreasing (for finite values).
//! * The index mapping is a valid permutation of `0..n`.
//! * Non-finite values maintain their relative insertion order at the end.
//!
//! ## Non-goals
//!
//! * This module does not perform data validation or LOWESS calculation.
//!
//! ## Visibility
//!
//! [`SortedData`] is internal to the engine but public for adapter access.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// Data Structures
// ============================================================================

/// Result of sorting input data by x-coordinates.
pub struct SortedData<T> {
    /// Sorted x-coordinates (finite values first).
    pub x: Vec<T>,

    /// Y-coordinates reordered to match sorted x-coordinates.
    pub y: Vec<T>,

    /// Index mapping where `indices[sorted_pos] = original_pos`.
    pub indices: Vec<usize>,
}

impl<T> SortedData<T> {
    /// Returns the number of points in the sorted data.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns `true` if the sorted data contains no points.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

// ============================================================================
// Sorting Functions
// ============================================================================

/// Sort input data by x-coordinates in ascending order.
///
/// 1. Pairs (x, y) with their original indices.
/// 2. Performs a stable sort:
///    - Finite values are ordered ascending.
///    - Non-finite values (NaN, Inf) are moved to the end.
/// 3. Extracts sorted arrays and permutation mapping.
pub fn sort_by_x<T: Float>(x: &[T], y: &[T]) -> SortedData<T> {
    // Create tuples of (x_value, y_value, original_index)
    let mut pairs: Vec<(T, T, usize)> = x
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&xi, &yi))| (xi, yi, i))
        .collect();

    // Stable sort to preserve order of equal x values for determinism
    pairs.sort_by(|a, b| {
        match (a.0.is_finite(), b.0.is_finite()) {
            (true, true) => {
                // Both finite: normal comparison
                a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
            }
            (true, false) => Ordering::Less, // Finite values come first
            (false, true) => Ordering::Greater, // Non-finite values at end
            (false, false) => {
                // Both non-finite: preserve original order
                a.2.cmp(&b.2)
            }
        }
    });

    // Extract sorted components
    SortedData {
        x: pairs.iter().map(|p| p.0).collect(),
        y: pairs.iter().map(|p| p.1).collect(),
        indices: pairs.iter().map(|p| p.2).collect(),
    }
}

/// Map sorted results back to the original input order in O(n) time.
pub fn unsort<T: Float>(sorted_values: &[T], indices: &[usize]) -> Vec<T> {
    let n = indices.len();
    let mut result = vec![T::zero(); n];

    // Map each sorted position back to its original position
    for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
        result[orig_idx] = sorted_values[sorted_idx];
    }

    result
}
