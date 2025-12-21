//! Layer 5: Engine
//!
//! Core execution logic for LOWESS smoothing.
//!
//! This layer orchestrates the smoothing process by coordinating between
//! primitives (traits, utilities) and algorithms (kernels, regression, robustness).
//! It provides the main iteration loops and convergence detection.
//!
//! # Module Organization
//!
//! - **executor**: Unified execution engine for all modes
//! - **validator**: Input and configuration validation rules
//! - **output**: Structured results (clean data, residuals, diagnostics)
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine ← You are here
//!   ↓
//! Layer 4: Evaluation (cv, diagnostics, intervals)
//!   ↓
//! Layer 3: Algorithms (regression, robustness, interpolation)
//!   ↓
//! Layer 2: Math (kernel, mad)
//!   ↓
//! Layer 1: Primitives (errors, traits, window)
//! ```

/// Unified execution engine for LOWESS smoothing.
///
/// Provides:
/// - High-level orchestration of the smoothing pipeline
/// - Support for all execution modes (Batch, Streaming, Online)
/// - Integration of regression and robustness steps
pub mod executor;

/// Validation utilities.
///
/// Provides:
/// - Checks for data consistency (NaNs, lengths)
/// - Configuration bound validation
/// - Shared validation logic for all adapters
pub mod validator;

/// Output types for LOWESS operations.
///
/// Provides:
/// - The `LowessResult` container struct
/// - Helpers for residuals and diagnostics formatting
/// - Standardized return type for all operations
pub mod output;
