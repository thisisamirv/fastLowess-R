//! Boundary padding utilities for reducing smoothing bias at data edges.
//!
//! This module implements various strategies for extending a dataset beyond its
//! original boundaries. By padding the input data, local regression at the
//! edges has more context, which helps mitigate the "boundary effect" where
//! the fit typically exhibits higher bias.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::primitives::partition::BoundaryPolicy;
use num_traits::Float;

/// Apply a boundary policy to pad the input data.
pub fn apply_boundary_policy<T: Float>(
    x: &[T],
    y: &[T],
    window_size: usize,
    policy: BoundaryPolicy,
) -> (Vec<T>, Vec<T>) {
    let n = x.len();
    if n < 2 || window_size < 2 {
        return (x.to_vec(), y.to_vec());
    }

    // Number of points to pad on each side (half-window)
    let pad_len = (window_size / 2).min(n - 1);
    if pad_len == 0 {
        return (x.to_vec(), y.to_vec());
    }

    let total_len = n + 2 * pad_len;
    let mut px = Vec::with_capacity(total_len);
    let mut py = Vec::with_capacity(total_len);

    // 1. Prepend padding
    match policy {
        BoundaryPolicy::Extend => {
            let x0 = x[0];
            let y0 = y[0];
            let dx = x[1] - x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - T::from(i).unwrap() * dx);
                py.push(y0);
            }
        }
        BoundaryPolicy::Reflect => {
            let x0 = x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - (x[i] - x0));
                py.push(y[i]);
            }
        }
        BoundaryPolicy::Zero => {
            let x0 = x[0];
            let dx = x[1] - x[0];
            for i in (1..=pad_len).rev() {
                px.push(x0 - T::from(i).unwrap() * dx);
                py.push(T::zero());
            }
        }
    }

    // 2. Add original data
    px.extend_from_slice(x);
    py.extend_from_slice(y);

    // 3. Append padding
    match policy {
        BoundaryPolicy::Extend => {
            let xn = x[n - 1];
            let yn = y[n - 1];
            let dx = x[n - 1] - x[n - 2];
            for i in 1..=pad_len {
                px.push(xn + T::from(i).unwrap() * dx);
                py.push(yn);
            }
        }
        BoundaryPolicy::Reflect => {
            let xn = x[n - 1];
            for i in 1..=pad_len {
                px.push(xn + (xn - x[n - 1 - i]));
                py.push(y[n - 1 - i]);
            }
        }
        BoundaryPolicy::Zero => {
            let xn = x[n - 1];
            let dx = x[n - 1] - x[n - 2];
            for i in 1..=pad_len {
                px.push(xn + T::from(i).unwrap() * dx);
                py.push(T::zero());
            }
        }
    }

    (px, py)
}
