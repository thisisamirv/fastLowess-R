# fastLowess (R binding for fastLowess Rust crate)

[![License](https://img.shields.io/badge/License-AGPL--3.0%20OR%20Commercial-blue.svg)](LICENSE)
[![R-CMD-check](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml/badge.svg)](https://github.com/thisisamirv/fastLowess-R/actions/workflows/ci.yml)

**High-performance LOWESS (Locally Weighted Scatterplot Smoothing) for R** — Significant speedup over `stats::lowess` with robust statistics, confidence intervals, and parallel execution. Built on the [fastLowess](https://github.com/thisisamirv/fastLowess) Rust crate.

## Why This Package?

- ⚡ **Blazingly Fast**: Sub-millisecond smoothing for 1000 points, significantly faster than base R
- 🎯 **Production-Ready**: Comprehensive error handling, numerical stability, extensive testing
- 📊 **Feature-Rich**: Confidence/prediction intervals, multiple kernels, cross-validation
- 🚀 **Scalable**: Parallel execution, streaming mode, delta optimization
- 🔬 **Scientific**: Validated against R implementation to machine precision

## Quick Start

```r
library(fastLowess)

x <- seq(1, 5, length.out = 5)
y <- c(2.0, 4.1, 5.9, 8.2, 9.8)

# Basic smoothing
result <- smooth(x, y, fraction = 0.5)

print(result$y)
# [1] 2.0 4.1 5.9 8.2 9.8
```

## Installation

You can install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("thisisamirv/fastLowess-R")
```

*(Note: Requires cargo/rustc to compile from source)*

## Features at a Glance

| Feature                  | Description                             | Use Case                      |
| ------------------------ | --------------------------------------- | ----------------------------- |
| **Robust Smoothing**     | IRLS with Bisquare/Huber/Talwar weights | Outlier-contaminated data     |
| **Confidence Intervals** | Point-wise standard errors & bounds     | Uncertainty quantification    |
| **Cross-Validation**     | Auto-select optimal fraction            | Unknown smoothing parameter   |
| **Multiple Kernels**     | Tricube, Epanechnikov, Gaussian, etc.   | Different smoothness profiles |
| **Parallel Execution**   | Multi-threaded via Rust/Rayon           | Large datasets (n > 1000)     |
| **Streaming Mode**       | Constant memory usage                   | Very large datasets           |

## Common Use Cases

### 1. Robust Smoothing (Handle Outliers)

```r
x <- seq(1, 5, length.out = 5)
y <- c(2.0, 4.1, 100.0, 8.2, 9.8)  # Outlier at index 3

# Use robust iterations to downweight outliers
result <- smooth(
    x, y,
    fraction = 0.7,
    iterations = 5L,  # Robust iterations
    return_robustness_weights = TRUE
)

# Check weights (low weight = outlier)
print(result$robustness_weights)
```

### 2. Uncertainty Quantification

```r
result <- smooth(
    x, y,
    fraction = 0.5,
    confidence_intervals = 0.95,
    prediction_intervals = 0.95
)

# Access confidence bands
head(data.frame(
    x = result$x,
    smooth = result$y,
    lower = result$confidence_lower,
    upper = result$confidence_upper
))
```

### 3. Automatic Parameter Selection (Cross-Validation)

```r
# Cross-validation is integrated into smooth()
result <- smooth(
    x, y,
    cv_fractions = c(0.2, 0.3, 0.5, 0.7),
    cv_method = "kfold",
    cv_k = 5L
)

cat("Optimal fraction:", result$fraction_used, "\n")
print(result$cv_scores)
```

### 4. Large Dataset Optimization (Streaming)

```r
# Streaming mode for very large datasets
# Keeps memory usage constant by processing in chunks
result <- smooth_streaming(
    x, y,
    fraction = 0.3,
    chunk_size = 5000L,
    overlap = 500L
)
```

### 5. Production Monitoring (Diagnostics)

```r
result <- smooth(
    x, y,
    fraction = 0.5,
    iterations = 3L,
    return_diagnostics = TRUE
)

# Access diagnostics list
print(result$diagnostics)
# $rmse, $mae, $r_squared, etc.
```

## Parameter Selection Guide

### Main Parameters

- **`fraction`**: Smoothing window size (0.0 - 1.0). Default `0.67` (Cleveland's choice).
- **`iterations`**: Robustness iterations. `0` for speed, `3` (default) for outlier resistance.
- **`weight_function`**: Kernel function. "tricube" (default), "gaussian", "epanechnikov".

## API Reference

### `smooth()`

Primary interface for batch smoothing.

```r
smooth(x, y, fraction = 0.67, iterations = 3L, ...)
```

### `smooth_streaming()`

Chunked processing for large datasets.

```r
smooth_streaming(x, y, chunk_size = 1000L, ...)
```

### `smooth_online()`

Sliding window for real-time/incremental data.

```r
smooth_online(x, y, window_capacity = 100L, ...)
```

## Demos

Run included demos to see the package in action:

```r
demo(package = "fastLowess")
demo("batch_smoothing")
```

## Performance Benchmarks

Comparison against Python `statsmodels` (pure Python/NumPy vs Rust extension). Note that `fastLowess-R` uses the same Rust backend, so performance is comparable.

| Dataset Size  | statsmodels | fastLowess | Speedup  |
| ------------- | ----------- | ---------- | -------- |
| 100 points    | 1.79 ms     | 0.13 ms    | **14×**  |
| 500 points    | 9.86 ms     | 0.26 ms    | **38×**  |
| 1,000 points  | 22.80 ms    | 0.39 ms    | **59×**  |
| 10,000 points | 742.99 ms   | 2.59 ms    | **287×** |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

See the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{fastLowess_R_2025,
  author = {Valizadeh, Amir},
  title = {fastLowess: High-performance LOWESS for R},
  year = {2025},
  url = {https://github.com/thisisamirv/fastLowess-R},
  version = {0.1.0}
}
```

## Author

**Amir Valizadeh**  
📧 <thisisamirv@gmail.com>  
🔗 [GitHub](https://github.com/thisisamirv/fastLowess-R)
