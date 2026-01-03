#' Common argument validation and coercion
#'
#' @description
#' Internal helper to validate x, y, fraction, and iterations inputs and
#' force them to the correct types for Rust FFI.
#'
#' @param x Numeric vector
#' @param y Numeric vector
#' @param fraction Numeric
#' @param iterations Integer
#'
#' @return A list containing the coerced x, y, fraction, and iterations.
#' @noRd
validate_common_args <- function(x, y, fraction, iterations) {
    if (length(x) != length(y)) {
        stop("x and y must have the same length")
    }
    if (length(x) < 3) {
        stop("At least 3 data points are required")
    }

    list(
        x = as.double(x),
        y = as.double(y),
        fraction = as.double(fraction),
        iterations = as.integer(iterations)
    )
}
