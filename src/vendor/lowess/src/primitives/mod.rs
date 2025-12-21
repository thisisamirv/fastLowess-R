//! Layer 1: Primitives
//!
//! Core building blocks and types.
//!
//! This layer provides the primitive abstractions, data structures, and
//! utility functions used throughout the crate. It has zero internal
//! dependencies within the crate.
//!
//! # Module Organization
//!
//! - **errors**: Shared error types (LowessError)
//! - **window**: Sliding window data structures
//! - **partition**: Utilities for data chunking/partitioning
//! - **sorting**: Low-level sorting helpers
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
//! Layer 2: Math (kernel, mad)
//!   ↓
//! Layer 1: Primitives ← You are here
//! ```

/// Sorting utilities.
///
/// Provides:
/// - Indices-based sorting
/// - Reversible sorting (unsort) operations
/// - Sort-by-x helpers
pub mod sorting;

/// Windowing logic.
///
/// Provides:
/// - Dynamic span calculation
/// - Nearest neighbor search
/// - Sliding window state management
pub mod window;

/// Partition and merge utilities for chunked data processing.
///
/// Provides:
/// - Data chunking logic
/// - Online update policies
/// - Boundary handling policies
pub mod partition;

/// Shared error types.
///
/// Provides:
/// - Unified `LowessError` enum
/// - Specific error variants
/// - Error propagation utilities
pub mod errors;
