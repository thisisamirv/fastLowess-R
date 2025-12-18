#' Streaming LOWESS for Large Datasets
#'
#' @description
#' Perform LOWESS smoothing using a streaming/chunked approach for large
#' datasets. Processes data in chunks to maintain constant memory usage,
#' suitable for datasets too large to fit in memory.
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction (default: 0.3). Lower values recommended
#'   for streaming to ensure good local fits within chunks.
#' @param chunk_size Number of points to process in each chunk (default: 5000).
#' @param overlap Number of points to overlap between chunks (default: 10
#'   percent of chunk_size). Overlap ensures smooth transitions between chunks.
#' @param iterations Number of robustness iterations (default: 3).
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param parallel Logical, whether to enable parallel chunk processing
#'   (default: TRUE).
#'
#' @return A list containing: x (sorted x values), y (smoothed y values),
#'   fraction_used.
#'
#' @examples
#' # Process a large dataset in chunks
#' n <- 50000
#' x <- seq(0, 100, length.out = n)
#' y <- sin(x) + rnorm(n, sd = 0.5)
#' result <- smooth_streaming(x, y, chunk_size = 5000L, overlap = 500L)
#'
#' @export
smooth_streaming <- function(x,
                             y,
                             fraction = 0.3,
                             chunk_size = 5000L,
                             overlap = NULL,
                             iterations = 3L,
                             weight_function = "tricube",
                             robustness_method = "bisquare",
                             parallel = TRUE) {
  # Validate inputs
  if (length(x) != length(y)) {
    stop("x and y must have the same length")
  }
  if (length(x) < 3) {
    stop("At least 3 data points are required")
  }

  # Ensure proper types
  x <- as.double(x)
  y <- as.double(y)
  fraction <- as.double(fraction)
  chunk_size <- as.integer(chunk_size)
  iterations <- as.integer(iterations)

  if (!is.null(overlap)) {
    overlap <- as.integer(overlap)
  }

  # Call the Rust function
  .Call(
    "wrap__smooth_streaming",
    x, y,
    fraction,
    chunk_size,
    overlap,
    iterations,
    weight_function,
    robustness_method,
    parallel,
    PACKAGE = "fastLowess"
  )
}
