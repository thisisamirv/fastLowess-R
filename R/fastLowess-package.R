#' fastLowess: High-Performance LOWESS Smoothing
#'
#' The fastLowess package provides high-performance LOWESS (Locally Weighted
#' Scatterplot Smoothing) using a Rust backend for speed and efficiency.
#'
#' @section Main Functions:
#' \itemize{
#'   \item \code{\link{smooth}}: Primary LOWESS interface for batch processing
#'   \item \code{\link{smooth_streaming}}: Chunked processing for large datasets
#'   \item \code{\link{smooth_online}}: Sliding window for real-time data
#' }
#'
#' @section Features:
#' \itemize{
#'   \item Multiple weight functions (tricube, gaussian, epanechnikov, etc.)
#'   \item Robustness iterations for outlier handling
#'   \item Confidence and prediction intervals
#'   \item Cross-validation for optimal parameter selection
#'   \item Parallel processing support
#' }
#'
#' @section Examples:
#' \preformatted{
#' # Run available demos:
#' demo(package = "fastLowess")
#'
#' # 1. Batch Smoothing Demo
#' # Demonstrates basic batch processing on a static dataset.
#' demo("batch_smoothing", package = "fastLowess", ask = FALSE)
#'
#' # 2. Online Smoothing Demo
#' # Shows sliding window smoothing simulating real-time data updates.
#' demo("online_smoothing", package = "fastLowess", ask = FALSE)
#'
#' # 3. Streaming Smoothing Demo
#' # Illustrates chunked processing for datasets larger than memory.
#' demo("streaming_smoothing", package = "fastLowess", ask = FALSE)
#' }
#'
#' @docType package
#' @name fastLowess-package
#' @useDynLib fastLowess, .registration = TRUE
NULL
