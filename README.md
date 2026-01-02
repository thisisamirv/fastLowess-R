# fastlowess

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![R-CMD-check](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml/badge.svg)](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml)

**High-performance parallel LOWESS (Locally Weighted Scatterplot Smoothing) for R** — A high-level wrapper around the [`fastLowess`](https://github.com/thisisamirv/fastLowess) Rust crate that adds rayon-based parallelism and seamless R integration.

## Features

- **Parallel by Default**: Multi-core regression fits via [rayon](https://crates.io/crates/rayon), achieving substantial speedups on large datasets.
- **Robust Statistics**: MAD-based scale estimation and IRLS with Bisquare, Huber, or Talwar weighting.
- **Uncertainty Quantification**: Point-wise standard errors, confidence intervals, and prediction intervals.
- **Optimized Performance**: Delta optimization for skipping dense regions and streaming/online modes.
- **Parameter Selection**: Built-in cross-validation for automatic smoothing fraction selection.
- **Production-Ready**: Comprehensive error handling, numerical stability, and high-performance numerical core.

## Robustness Advantages

Built on the same core as `lowess`, this implementation is **more robust than statsmodels** due to two key design choices:

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

The `fastlowess` R package demonstrates massive performance gains over R's `stats::lowess`. The benchmarks compare `fastlowess` (both Serial and Parallel execution modes) against this standard implementation.

The results show that `fastlowess` is the decisive winner across all benchmarks, achieving an **average speedup of 280x** and a **maximum speedup of 1169x**.

The table below shows speedups relative to the **statsmodels baseline**.

| Name                  | statsmodels |      R      |  fastlowess   |
|-----------------------|-------------|-------------|---------------|
| clustered             |  162.77ms   |  [82.8x]²   |  [170-432x]¹  |
| constant_y            |  133.63ms   |  [92.3x]²   |  [176-372x]¹  |
| delta_large           |   0.51ms    |   [0.8x]²   |  [3.3-2.0x]¹  |
| delta_medium          |   0.79ms    |   [1.3x]²   |  [3.7-3.3x]¹  |
| delta_none            |  414.86ms   |   [2.5x]²   |  [3.2-16x]¹   |
| delta_small           |   1.45ms    |   [1.7x]²   |  [3.6-4.4x]¹  |
| extreme_outliers      |  488.96ms   |  [106.4x]²  |  [168-373x]¹  |
| financial_1000        |   13.55ms   |  [76.6x]²   |  [135-105x]¹  |
| financial_10000       |  302.20ms   |  [168.3x]²  |  [379-480x]¹  |
| financial_500         |   6.49ms    |  [58.0x]¹   |   [92-54x]²   |
| financial_5000        |  103.94ms   |  [117.3x]²  |  [252-336x]¹  |
| fraction_0.05         |  122.00ms   |  [177.6x]²  |  [376-274x]¹  |
| fraction_0.1          |  140.59ms   |  [112.8x]²  |  [252-219x]¹  |
| fraction_0.2          |  181.57ms   |  [85.3x]²   |  [180-283x]¹  |
| fraction_0.3          |  220.98ms   |  [84.8x]²   |  [151-304x]¹  |
| fraction_0.5          |  296.47ms   |  [80.9x]²   |  [125-366x]¹  |
| fraction_0.67         |  362.59ms   |  [83.1x]²   |  [115-428x]¹  |
| genomic_1000          |   17.82ms   |  [15.9x]²   |   [16-23x]¹   |
| genomic_10000         |  399.90ms   |   [3.6x]²   |  [4.5-18x]¹   |
| genomic_5000          |  138.49ms   |   [5.0x]²   |  [6.1-21x]¹   |
| genomic_50000         |  6776.57ms  |   [2.4x]²   |  [3.1-12x]¹   |
| high_noise            |  435.85ms   |  [132.6x]²  |  [118-381x]¹  |
| iterations_0          |   45.18ms   |  [128.4x]²  |  [212-497x]¹  |
| iterations_1          |   94.10ms   |  [114.3x]²  |  [195-460x]¹  |
| iterations_10         |  495.65ms   |  [116.0x]²  |  [172-428x]¹  |
| iterations_2          |  135.48ms   |  [109.0x]²  |  [180-399x]¹  |
| iterations_3          |  181.56ms   |  [108.8x]²  |  [178-408x]¹  |
| iterations_5          |  270.58ms   |  [110.4x]²  |  [174-356x]¹  |
| scale_1000            |   17.95ms   |  [82.6x]¹   |  [131-51x]²   |
| scale_10000           |  408.13ms   |  [178.1x]²  |  [378-270x]¹  |
| scale_5000            |  139.81ms   |  [133.6x]²  |  [254-224x]¹  |
| scale_50000           |  6798.58ms  |  [661.0x]²  | [987-1169x]¹  |
| scientific_1000       |   19.04ms   |  [70.1x]²   |  [103-75x]¹   |
| scientific_10000      |  479.57ms   |  [190.7x]²  |  [316-461x]¹  |
| scientific_500        |   8.59ms    |  [49.6x]¹   |   [69-45x]²   |
| scientific_5000       |  161.42ms   |  [124.9x]²  |  [205-273x]¹  |
| scale_100000**        |      -      |      -      |    1-1.5x     |

\* **fastlowess**: Shows speedup range `[Serial-Parallel]`. E.g., `[12-48x]` means 12x speedup (Sequential) and 48x speedup (Parallel).

\*\* **Large Scale**: `fastlowess (Serial)` is the baseline (1x).

¹ Winner (Fastest implementation)

² Runner-up (Second fastest implementation)

**Key Takeaways**::

1. **Dominant Performance**: `fastlowess` is consistently the fastest implementation. Even in **Serial** mode, it significantly outperforms `stats::lowess`.
2. **Parallel Scaling**:
    - **Large Datasets**: Parallel execution provides massive gains. For example, `scale_50000` shows a jump from ~987x (Serial) to ~1169x (Parallel) speedup.
    - **Small Datasets**: For very small datasets (e.g., `scale_1000`, `financial_500`), Serial execution is often faster than Parallel due to thread overhead (e.g., `[131-51x]`).
3. **R vs Statsmodels**: `R` is a strong runner-up, generally ~80-150x faster than `statsmodels`, but `fastlowess` extends this lead further.
4. **Handling Complex Cases**: `fastlowess` maintains its performance advantage even in pathological cases like `high_noise` and `extreme_outliers`.

Check [Benchmarks for fastLowess](https://github.com/thisisamirv/fastLowess-R/tree/bench/benchmarks) for detailed results and reproducible benchmarking code.

## Validation

The `fastlowess` package is a **numerical twin** of R's `lowess` implementation:

| Aspect          | Status         | Details                                    |
|-----------------|----------------|--------------------------------------------|
| **Accuracy**    | ✅ EXACT MATCH | Max diff < 1e-12 across all scenarios      |
| **Consistency** | ✅ PERFECT     | 15/15 scenarios pass with strict tolerance |
| **Robustness**  | ✅ VERIFIED    | Robust smoothing matches R exactly         |

Check [Validation](https://github.com/thisisamirv/fastLowess-R/tree/bench/validation) for detailed scenario results.

## Installation

Install pre-built binaries from R-universe (no Rust required):

```r
install.packages("fastlowess", repos = "https://thisisamirv.r-universe.dev")
```

### From Source (Development)

To install from source (requires [Rust](https://rustup.rs/)):

```r
# install.packages("devtools")
devtools::install_github("thisisamirv/fastLowess-R")
```

**Note**: First-time source installation may take 2-5 minutes to compile.

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

    # Robustness iterations
    iterations = 3L,

    # Interpolation threshold
    delta = 0.01,

    # Kernel function
    weight_function = "tricube",

    # Robustness method
    robustness_method = "bisquare",

    # Zero-weight fallback
    zero_weight_fallback = "use_local_mean",

    # Boundary handling
    boundary_policy = "extend",

    # Intervals
    confidence_intervals = 0.95,
    prediction_intervals = 0.95,

    # Diagnostics
    return_diagnostics = TRUE,
    return_residuals = TRUE,
    return_robustness_weights = TRUE,

    # Cross-validation
    cv_fractions = c(0.3, 0.5, 0.7),
    cv_method = "kfold",
    cv_k = 5L,

    # Convergence
    auto_converge = 1e-4,

    # Parallelism
    parallel = TRUE
)
```

## Result Structure

The `smooth()` function returns a named list:

```r
result$x                    # Sorted independent variable values
result$y                    # Smoothed dependent variable values
result$standard_errors      # Point-wise standard errors
result$confidence_lower     # Lower bound of confidence interval
result$confidence_upper     # Upper bound of confidence interval
result$prediction_lower     # Lower bound of prediction interval
result$prediction_upper     # Upper bound of prediction interval
result$residuals            # Residuals (y - fit)
result$robustness_weights   # Final robustness weights
result$diagnostics          # Diagnostics (RMSE, R², etc.)
result$iterations_used      # Number of iterations performed
result$fraction_used        # Smoothing fraction used
result$cv_scores            # CV scores for each candidate
```

## Streaming Processing

For datasets that don't fit in memory:

```r
result <- smooth_streaming(
    x, y,
    fraction = 0.3,
    chunk_size = 5000L,
    overlap = 500L,
    parallel = TRUE
)
```

## Online Processing

For real-time data streams:

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

- **0.1-0.3**: Local, captures rapid changes
- **0.4-0.6**: Balanced, general-purpose
- **0.7-1.0**: Global, smooth trends only
- **Default: 0.67** (2/3, Cleveland's choice)

### Robustness Iterations

- **0**: Clean data, speed critical
- **1-3**: Default, good balance
- **4-5**: Heavy outliers

### Kernel Function

- **Tricube** (default): Best all-around
- **Epanechnikov**: Optimal MSE
- **Gaussian**: Very smooth
- **Uniform**: Moving average

### Delta Optimization

- **None**: Small datasets (n < 1000)
- **0.01 × range(x)**: Good starting point for dense data
- **Manual tuning**: Adjust based on data density

## Demos

Check the included demos:

```r
demo(package = "fastlowess")
demo("batch_smoothing")
demo("online_smoothing")
demo("streaming_smoothing")
```

## Related Work

- [fastLowess (Rust core)](https://github.com/thisisamirv/fastLowess)
- [fastLowess-py (Python wrapper)](https://github.com/thisisamirv/fastLowess-py)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of

- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## References

- Cleveland, W.S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots". *JASA*.
- Cleveland, W.S. (1981). "LOWESS: A Program for Smoothing Scatterplots". *The American Statistician*.
