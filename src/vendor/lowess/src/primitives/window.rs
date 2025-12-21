//! Windowing primitives for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides low-level data structures for managing sliding windows
//! over sorted datasets, ensuring each local regression uses the nearest neighbors.
//!
//! ## Design notes
//!
//! * **Pre-sorted**: Algorithms assume the independent variable (x) is non-decreasing.
//! * **Nearest Neighbors**: Windows slide to find the q points closest to the target.
//! * **Boundary Handling**: Adjusts window placement to remain within bounds near edges.
//! * **Zero-Copy**: Operates as a simple index-based view into the underlying data.
//!
//! ## Key concepts
//!
//! ### Window Lifecycle
//! 1. **Span Calculation**: `calculate_span` determines the window size q from the fraction.
//! 2. **Initialization**: `initialize` sets the starting bounds for the first point.
//! 3. **Recentering**: `recenter` slides the bounds for subsequent points in O(n) total time.
//!
//! ## Invariants
//!
//! * A valid window satisfies `left <= right`.
//! * Window indices always remain within the caller-provided array bounds.
//! * The window correctly captures the q nearest neighbors for a given target point.
//!
//! ## Visibility
//!
//! This module is an internal detail used by the execution engine.

use num_traits::Float;

/// Inclusive window bounds `[left, right]` for a local fit.
#[derive(Copy, Clone, Debug)]
pub struct Window {
    /// Left boundary index (inclusive).
    pub left: usize,

    /// Right boundary index (inclusive).
    pub right: usize,
}

impl Window {
    /// Create a new window if `left <= right`.
    #[inline]
    pub fn new(left: usize, right: usize) -> Option<Self> {
        if left <= right {
            Some(Self { left, right })
        } else {
            None
        }
    }

    /// Returns the number of points in the window.
    #[inline]
    pub fn len(&self) -> usize {
        if self.left <= self.right {
            self.right - self.left + 1
        } else {
            0
        }
    }

    /// Returns `true` if the window contains no points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ========================================================================
    // Window Management
    // ========================================================================

    /// Initialize window boundaries for the first point in a sequence.
    pub fn initialize(idx: usize, window_size: usize, n: usize) -> Self {
        debug_assert!(
            window_size >= 1,
            "initialize_window: window_size must be at least 1"
        );

        if window_size >= n {
            return Self {
                left: 0,
                right: n.saturating_sub(1),
            };
        }

        let half = window_size / 2;
        let mut left = idx.saturating_sub(half);
        let max_left = n - window_size;
        if left > max_left {
            left = max_left;
        }

        let right = left + window_size - 1;
        Self { left, right }
    }

    /// Update boundaries to maintain nearest-neighbor centering.
    pub fn recenter<T: Float>(&mut self, x: &[T], current: usize, n: usize) {
        debug_assert!(current < n, "recenter: current index out of bounds");

        if current >= n || n == 0 {
            return;
        }

        self.left = self.left.min(n - 1);
        self.right = self.right.min(n - 1);

        let x_current = x[current];

        // Search for the optimal window position (nearest neighbors)
        while self.right < n - 1 {
            let d_left = x_current - x[self.left];
            let d_right = x[self.right + 1] - x_current;

            if d_left <= d_right {
                break;
            }

            self.left += 1;
            self.right += 1;
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Compute the maximum distance from `x_current` to any point in the window.
    pub fn max_distance<T: Float>(&self, x: &[T], x_current: T) -> T {
        T::max(x_current - x[self.left], x[self.right] - x_current)
    }

    /// Calculate window size q from fraction alpha and data length n.
    #[inline]
    pub fn calculate_span<T: Float>(n: usize, frac: T) -> usize {
        let frac_n = frac * T::from(n).unwrap();
        let frac_n_int = frac_n.to_usize().unwrap_or(0);
        usize::max(2, usize::min(n, frac_n_int))
    }
}
