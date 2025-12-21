//! Layer 3: Algorithms
//!
//! Core LOWESS algorithms.
//!
//! This layer implements the core logic for local weighted regression, robustness
//! iterations, and interpolation. It contains the "business logic" of LOWESS
//! but is orchestrated by the engine layer.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine (executor, output, validator)
//!   ↓
//! Layer 4: Evaluation (cv, diagnostics, intervals)
//!   ↓
//! Layer 3: Algorithms ← You are here
//!   ↓
//! Layer 2: Math (kernel, mad)
//!   ↓
//! Layer 1: Primitives (errors, traits, window)
//! ```

/// Local weighted regression implementations.
///
/// Provides:
/// - Linear and polynomial regression
/// - Weighted least squares (WLS) optimization
/// - Point-wise neighborhood fitting
pub mod regression;

/// Robustness weight updates for outlier downweighting.
///
/// Provides:
/// - Bisquare function implementation
/// - Iterative Reweighted Least Squares (IRLS) logic
/// - Outlier detection and weight adjustment
pub mod robustness;

/// Interpolation and delta optimization utilities.
///
/// Provides:
/// - Delta-based optimization for speed
/// - Linear interpolation between anchor points
/// - Skipped computation for dense regions
pub mod interpolation;
