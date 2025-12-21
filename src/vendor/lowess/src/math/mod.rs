//! Layer 2: Math
//!
//! Pure mathematical functions.
//!
//! This layer provides pure mathematical functions used throughout LOWESS:
//! - Kernel functions for distance-based weighting
//! - Robust statistics (MAD)
//!
//! These are reusable mathematical building blocks with no algorithm-specific logic.
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
//! Layer 3: Algorithms (regression, robustness, interpolation)
//!   ↓
//! Layer 2: Math ← You are here
//!   ↓
//! Layer 1: Primitives (errors, traits, window)
//! ```

/// Kernel (weight) functions for distance-based weighting.
///
/// Provides:
/// - Standard weight functions (Tricube, Gaussian, Uniform, etc.)
/// - Distance normalization and scaling
/// - Weight calculation logic
pub mod kernel;

/// Median Absolute Deviation (MAD) computation.
///
/// Provides:
/// - Robust variance estimation
/// - Median calculation utilities
/// - Scale factor adjustment for normal consistency
pub mod mad;

/// Boundary padding utilities.
///
/// Provides:
/// - Padding strategies (Extend, Reflect, Zero)
/// - Coordination for reduce edge bias
pub mod boundary;
