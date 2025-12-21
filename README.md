# fastlowess

[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![R-CMD-check](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml/badge.svg)](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for R** — A high-level wrapper around the [`fastLowess`](https://github.com/thisisamirv/fastLowess) Rust crate that offers significant speedups over `stats::lowess` while providing robust statistics, uncertainty quantification, and memory-efficient streaming.

## Features

- **Parallel Execution**: Multi-core regression fits via Rust's Rayon, achieving substantial speedups on large datasets.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Production-Ready**: Comprehensive error handling, numerical stability, and high-performance numerical core.

## Robustness Advantages

This implementation is **more robust than statsmodels** due to two key design choices:

### MAD-Based Scale Estimation

For robustness weight calculations, this crate uses **Median Absolute Deviation (MAD)** for scale estimation:

```text
s = median(|r_i - median(r)|)
```

In contrast, statsmodels uses median of absolute residuals:

```text
s = median(|r_i|)
```

**Why MAD is more robust:**

- MAD is a **breakdown-point-optimal** estimator—it remains valid even when up to 50% of data are outliers.
- The median-centering step removes asymmetric bias from residual distributions.
- MAD provides consistent outlier detection regardless of whether residuals are centered around zero.

### Boundary Padding

This crate applies **boundary policies** (Extend, Reflect, Zero) at dataset edges:

- **Extend**: Repeats edge values to maintain local neighborhood size.
- **Reflect**: Mirrors data symmetrically around boundaries.
- **Zero**: Pads with zeros (useful for signal processing).

statsmodels does not apply boundary padding, which can lead to:

- Biased estimates near boundaries due to asymmetric local neighborhoods.
- Increased variance at the edges of the smoothed curve.

### Gaussian Consistency Factor

For interval estimation (confidence/prediction), residual scale is computed using:

```text
sigma = 1.4826 * MAD
```

The factor 1.4826 = 1/Phi^-1(3/4) ensures consistency with the standard deviation under Gaussian assumptions.

## Performance Advantages

Benchmarked against Python's `statsmodels`. Achieves **50-1400× faster performance** across different tested scenarios.

### Summary

| Category         | Matched | Median Speedup | Mean Speedup |
|------------------|---------|----------------|--------------|
| **Scalability**  | 5       | **236×**       | 528×         |
| **Financial**    | 4       | **128×**       | 144×         |
| **Iterations**   | 6       | **104×**       | 106×         |
| **Pathological** | 4       | **107×**       | 97×          |
| **Scientific**   | 4       | **97×**        | 108×         |
| **Fraction**     | 6       | **92×**        | 119×         |
| **Genomic**      | 4       | **3.4×**       | 5×           |
| **Delta**        | 4       | **2×**         | 2.1×         |

### Top 10 Performance Wins

| Benchmark        | statsmodels | Rust   | Speedup   |
|------------------|-------------|--------|-----------|
| scale_100000     | 43.7s       | 31.0ms | **1409×** |
| scale_50000      | 11.2s       | 15.2ms | **734×**  |
| fraction_0.05    | 197.2ms     | 0.76ms | **258×**  |
| financial_10000  | 497.1ms     | 2.10ms | **237×**  |
| scale_10000      | 663.1ms     | 2.81ms | **236×**  |
| scientific_10000 | 777.2ms     | 4.24ms | **183×**  |
| fraction_0.1     | 227.9ms     | 1.40ms | **163×**  |
| scale_5000       | 229.9ms     | 1.42ms | **162×**  |
| financial_5000   | 170.9ms     | 1.05ms | **162×**  |
| scientific_5000  | 268.5ms     | 2.13ms | **126×**  |

Check [Benchmarks](https://github.com/thisisamirv/fastLowess-R/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Installation

### Prerequisites

This package requires **Rust** to compile from source.

1. **Install Rust**: Visit [rustup.rs](https://rustup.rs/) and follow instructions.
2. **Verify**: Run `rustc --version` in your terminal.

### Installing the Package

Once Rust is installed, you can install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("thisisamirv/fastLowess-R")
```

**Note**: First-time installation may take 2-5 minutes as it compiles the Rust core.

## Quick Start

```r
library(fastlowess)

x <- seq(1, 10, length.out = 100)
y <- sin(x) + rnorm(100, sd = 0.2)

# Basic smoothing (parallel by default)
result <- smooth(x, y, fraction = 0.3)

plot(x, y, main = "fastlowess Smoothing")
lines(result$x, result$y, col = "red", lwd = 2)
```

## Smoothing Parameters

```r
library(fastlowess)

smooth(
    x, y,
    # Smoothing span (0, 1]
    fraction = 0.5,

    # Robustness iterations for outlier resistance
    iterations = 3L,

    # Interpolation threshold for performance optimization
    # NULL (default) auto-calculates based on data range
    delta = 0.01,

    # Kernel function selection
    # Options: "tricube", "gaussian", "epanechnikov", "uniform", etc.
    weight_function = "tricube",

    # Robustness method selection
    # Options: "bisquare", "huber", "talwar"
    robustness_method = "bisquare",

    # Zero-weight fallback behavior
    # Options: "uselocalmean", "returnoriginal", "returnnone"
    zero_weight_fallback = "uselocalmean",

    # Boundary handling (for edge effects)
    # Options: "extend", "reflect", "zero"
    boundary_policy = "extend",

    # Uncertainty Quantification
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,

    # Output selection
    return_diagnostics = TRUE,
    return_residuals = TRUE,
    return_robustness_weights = TRUE,

    # Cross-validation (for automatic parameter selection)
    cv_fractions = c(0.3, 0.5, 0.7),
    cv_method = "kfold",
    cv_k = 5L,

    # Multi-threading (via Rust/Rayon)
    parallel = TRUE
)
```

## Result Structure

The `smooth()` function returns a named list containing:

```r
result <- list(
    x = ...,                   # Sorted independent variable values
    y = ...,                   # Smoothed dependent variable values
    standard_errors = ...,     # Point-wise standard errors (if computed)
    confidence_lower = ...,    # Lower bound of confidence interval
    confidence_upper = ...,    # Upper bound of confidence interval
    prediction_lower = ...,    # Lower bound of prediction interval
    prediction_upper = ...,    # Upper bound of prediction interval
    residuals = ...,           # Model residuals (y - y_fit)
    robustness_weights = ...,  # Final weights used for outlier handling
    diagnostics = list(...),   # Detailed fit diagnostics (RMSE, R^2, etc.)
    iterations_used = ...,     # Number of robustness iterations performed
    fraction_used = ...,       # Smoothing fraction used (best if via CV)
    cv_scores = ...            # RMSE scores for each CV candidate
)
```

## Execution Modes

### Streaming Processing

For datasets too large to fit in memory (processes in chunks):

```r
result <- smooth_streaming(
    x, y,
    fraction = 0.3,
    chunk_size = 5000L,
    overlap = 500L,
    parallel = TRUE
)
```

### Online Processing

For real-time data streams or sliding windows:

```r
result <- smooth_online(
    x, y,
    fraction = 0.2,
    window_capacity = 100L,
    update_mode = "incremental" # or "full"
)
```

## Parameter Selection Guide

### Fraction (Smoothing Span)

- **0.1-0.3**: Local, captures rapid changes (wiggly)
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)
- **Use CV** when uncertain

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-2**: Light contamination
- **3**: Default, good balance (recommended)
- **4-5**: Heavy outliers
- **>5**: Diminishing returns

### Kernel Function

- **Tricube** (default): Best all-around, smooth, efficient
- **Epanechnikov**: Theoretically optimal MSE
- **Gaussian**: Very smooth, no compact support
- **Uniform**: Fastest, least smooth (moving average)

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Demos

Run included demos to see the package in action:

```r
demo(package = "fastlowess")
demo("batch_smoothing")
demo("online_smoothing")
demo("streaming_smoothing")
```

## Validation

Validated against:

- **Base R (stats::lowess)**: Results matched to machine precision.
- **Original Paper**: Reproduces Cleveland (1979) results.

Check [Validation](https://github.com/thisisamirv/fastLowess-R/tree/bench/validation) for more information. Small variations in results are expected due to differences in scale estimation and padding.

## Related Work

- [fastLowess (Rust core)](https://github.com/thisisamirv/fastLowess)
- [fastLowess-py (Python wrapper)](https://github.com/thisisamirv/fastlowess-py)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual-licensed under **AGPL-3.0** (Open Source) or **Commercial License**.
Contact `<thisisamirv@gmail.com>` for commercial inquiries.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
