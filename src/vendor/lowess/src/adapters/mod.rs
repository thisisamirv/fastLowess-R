//! Layer 6: Adapters
//!
//! High-level execution adapters for LOWESS smoothing.
//!
//! This layer provides user-facing APIs that adapt the engine layer for different
//! execution modes and use cases:
//!
//! - **Batch**: Unified adapter for sequential execution
//! - **Streaming**: Chunked processing for large datasets
//! - **Online**: Incremental updates for real-time data
//!
//! # Choosing an Adapter
//!
//! - **Batch**: Default choice for most datasets
//! - **Streaming**: Very large datasets that don't fit in memory
//! - **Online**: Real-time data streams requiring incremental updates
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters ← You are here
//!   ↓
//! Layer 5: Engine (executor, output, validator)
//!   ↓
//! Layer 4: Evaluation (cv, diagnostics, intervals)
//!   ↓
//! Layer 3: Algorithms (regression, robustness, interpolation)
//!   ↓
//! Layer 2: Math (kernel, mad)
//!   ↓
//! Layer 1: Primitives (errors, traits, window)
//! ```

/// Unified batch adapter for LOWESS smoothing.
///
/// Provides:
/// - Sequential execution (always available)
/// - Cross-validation for bandwidth selection
/// - Automatic convergence detection
pub mod batch;

/// Streaming LOWESS for large datasets.
///
/// Provides:
/// - Chunked processing with configurable overlap
/// - Multiple merge strategies for chunk boundaries
/// - Memory-efficient processing of large datasets
pub mod streaming;

/// Online LOWESS for real-time data streams.
///
/// Provides:
/// - Incremental updates as new points arrive
/// - Sliding window with configurable capacity
/// - Low-latency per-point smoothing
pub mod online;
