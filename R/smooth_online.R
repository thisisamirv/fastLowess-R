#' Online LOWESS with Sliding Window
#'
#' @description
#' Perform LOWESS smoothing using an online/sliding window approach for
#' real-time data streams. Maintains a sliding window of recent points for
#' incremental updates without reprocessing the entire dataset.
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction (default: 0.2). Lower values recommended
#'   for online processing with small windows.
#' @param window_capacity Maximum number of points to retain in the sliding
#'   window (default: 100).
#' @param min_points Minimum number of points required before smoothing starts
#'   (default: 3). Points before this threshold use original y values.
#' @param iterations Number of robustness iterations (default: 3).
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param parallel Logical, whether to enable parallel processing
#'   (default: FALSE, as online mode typically processes sequentially).
#'
#' @return A list containing: x (input x values), y (smoothed y values),
#'   fraction_used.
#'
#' @examples
#' # Real-time sensor data smoothing
#' x <- 1:100
#' y <- sin(x / 10) + rnorm(100, sd = 0.3)
#' result <- smooth_online(x, y, window_capacity = 20L, min_points = 5L)
#'
#' @export
smooth_online <- function(x,
                          y,
                          fraction = 0.2,
                          window_capacity = 100L,
                          min_points = 3L,
                          iterations = 3L,
                          weight_function = "tricube",
                          robustness_method = "bisquare",
                          parallel = FALSE) {
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
  window_capacity <- as.integer(window_capacity)
  min_points <- as.integer(min_points)
  iterations <- as.integer(iterations)

  # Call the Rust function
  .Call(
    "wrap__smooth_online",
    x, y,
    fraction,
    window_capacity,
    min_points,
    iterations,
    weight_function,
    robustness_method,
    parallel,
    PACKAGE = "fastLowess"
  )
}
