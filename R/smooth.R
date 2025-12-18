#' LOWESS Smoothing with Batch Adapter
#'
#' @description
#' Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the input data.
#' This is the primary interface for LOWESS smoothing, processing the entire
#' dataset in memory with optional parallel execution.
#'
#' @param x Numeric vector of independent variable values.
#' @param y Numeric vector of dependent variable values (same length as x).
#' @param fraction Smoothing fraction, the proportion of data to use for each
#'   local regression (default: 0.67). Values between 0 and 1.
#' @param iterations Number of robustness iterations for outlier handling
#'   (default: 3). Use 0 for no robustness.
#' @param delta Interpolation optimization threshold. Points closer than delta
#'   will use linear interpolation instead of full regression. NULL (default)
#'   auto-calculates based on data range.
#' @param weight_function Kernel function for distance weighting. Options:
#'   "tricube" (default), "epanechnikov", "gaussian", "uniform", "biweight",
#'   "triangle", "cosine".
#' @param robustness_method Method for computing robustness weights. Options:
#'   "bisquare" (default), "huber", "talwar".
#' @param confidence_intervals Confidence level for confidence intervals
#'   (e.g., 0.95 for 95 percent CI). NULL (default) disables.
#' @param prediction_intervals Confidence level for prediction intervals
#'   (e.g., 0.95 for 95 percent PI). NULL (default) disables.
#' @param return_diagnostics Logical, whether to compute fit quality metrics
#'   (RMSE, MAE, R-squared, etc.). Default: FALSE.
#' @param return_residuals Logical, whether to include residuals in output.
#'   Default: FALSE.
#' @param return_robustness_weights Logical, whether to include final
#'   robustness weights in output. Default: FALSE.
#' @param zero_weight_fallback Fallback strategy when all weights are zero.
#'   Options: "use_local_mean" (default), "return_original", "return_none".
#' @param auto_converge Tolerance for automatic convergence detection. NULL
#'   (default) disables auto-convergence.
#' @param max_iterations Maximum number of robustness iterations when
#'   auto_converge is enabled. Default: 20.
#' @param cv_fractions Numeric vector of fractions to test for cross-validation.
#'   NULL (default) disables cross-validation.
#' @param cv_method Cross-validation method. Options: "kfold" (default), "loocv"
#'   (leave-one-out).
#' @param cv_k Number of folds for k-fold cross-validation (default: 5).
#'
#' @return A list containing: x (sorted x values), y (smoothed y values),
#'   fraction_used, and optionally: standard_errors, confidence_lower,
#'   confidence_upper, prediction_lower, prediction_upper, residuals,
#'   robustness_weights, iterations_used, cv_scores, diagnostics.
#'
#' @examples
#' # Basic smoothing
#' x <- 1:100
#' y <- sin(x / 10) + rnorm(100, sd = 0.2)
#' result <- smooth(x, y, fraction = 0.3)
#' plot(x, y)
#' lines(result$x, result$y, col = "red", lwd = 2)
#'
#' @export
smooth <- function(x,
                   y,
                   fraction = 0.67,
                   iterations = 3L,
                   delta = NULL,
                   weight_function = "tricube",
                   robustness_method = "bisquare",
                   confidence_intervals = NULL,
                   prediction_intervals = NULL,
                   return_diagnostics = FALSE,
                   return_residuals = FALSE,
                   return_robustness_weights = FALSE,
                   zero_weight_fallback = "use_local_mean",
                   auto_converge = NULL,
                   max_iterations = NULL,
                   cv_fractions = NULL,
                   cv_method = "kfold",
                   cv_k = 5L) {
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
  iterations <- as.integer(iterations)
  cv_k <- as.integer(cv_k)

  # Call the Rust function
  .Call(
    "wrap__smooth",
    x, y,
    fraction,
    iterations,
    delta,
    weight_function,
    robustness_method,
    confidence_intervals,
    prediction_intervals,
    return_diagnostics,
    return_residuals,
    return_robustness_weights,
    zero_weight_fallback,
    auto_converge,
    max_iterations,
    cv_fractions,
    cv_method,
    cv_k,
    PACKAGE = "fastLowess"
  )
}
