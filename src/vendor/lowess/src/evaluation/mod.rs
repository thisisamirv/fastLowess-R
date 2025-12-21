//! Layer 4: Evaluation
//!
//! Post-processing and model evaluation.
//!
//! This layer calculates high-level statistical metrics based on the smoothing results:
//! - Cross-validation for parameter selection
//! - Diagnostic metrics for fit quality
//! - Confidence and prediction intervals
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
//! Layer 4: Evaluation ← You are here
//!   ↓
//! Layer 3: Algorithms (regression, robustness, interpolation)
//!   ↓
//! Layer 2: Math (kernel, mad)
//!   ↓
//! Layer 1: Primitives (errors, traits, window)
//! ```

/// Cross-validation for bandwidth selection.
///
/// Provides:
/// - K-fold and Leave-One-Out validation
/// - Automated parameter tuning
/// - CV score calculation (MSE/RMSE)
pub mod cv;

/// Diagnostic metrics for fit quality assessment.
///
/// Provides:
/// - Residual statistics
/// - Goodness-of-fit metrics (R^2, RMSE, MAE)
/// - Effective degrees of freedom calculation
pub mod diagnostics;

/// Confidence and prediction interval computation.
///
/// Provides:
/// - Parametric confidence intervals
/// - Prediction intervals accounting for new observations
/// - Standard error estimation
pub mod intervals;
