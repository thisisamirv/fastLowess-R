# Benchmarks

## Parallel Execution (fastLowess)

### High-Level Summary

The `fastLowess` package with parallel execution demonstrates **5-287× faster performance** than Python's statsmodels across typical workloads. The implementation leverages rayon for parallel smoothing, achieving significant speedups especially for larger datasets and realistic scenarios.

| Category              | Median Speedup | Mean Speedup | Notes                          |
|-----------------------|----------------|--------------|--------------------------------|
| Basic Smoothing       | 58.7×          | 101.9×       | Scales massively with size     |
| Fraction Variations   | 39.5×          | 40.1×        | Consistent across all fractions|
| Robustness Iterations | 40.0×          | 40.1×        | Consistent across all iters    |
| Pathological Cases    | 54.0×          | 68.6×        | Excellent edge-case handling   |
| Realistic Scenarios   | 34.5×          | 38.5×        | Excellent real-world perf      |
| Delta Parameter       | 7.5×           | 7.6×         | Properly optimized             |

### Top Performance Wins

| Benchmark             | statsmodels   | fastLowess | Speedup    |
|-----------------------|---------------|------------|------------|
| basic_smoothing_10000 | 742.99 ms     | 2.59 ms    | **286.6×** |
| clustered_x           | 19.36 ms      | 0.16 ms    | **124.0×** |
| basic_smoothing_5000  | 229.76 ms     | 2.04 ms    | **112.4×** |
| financial_timeseries  | 14.73 ms      | 0.20 ms    | **72.0×**  |
| high_noise            | 28.84 ms      | 0.46 ms    | **62.6×**  |
| basic_smoothing_1000  | 22.80 ms      | 0.39 ms    | **58.7×**  |
| fraction_0.1          | 19.31 ms      | 0.36 ms    | **53.3×**  |
| iterations_10         | 62.81 ms      | 1.26 ms    | **49.8×**  |
| fraction_0.67         | 29.25 ms      | 0.63 ms    | **46.2×**  |
| extreme_outliers      | 34.94 ms      | 0.77 ms    | **45.4×**  |

### Detailed Results by Category

#### Basic Smoothing

| Dataset Size | statsmodels | fastLowess | Speedup  |
|--------------|-------------|------------|----------|
| 100          | 1.79 ms     | 0.13 ms    | 13.6×    |
| 500          | 9.86 ms     | 0.26 ms    | 38.3×    |
| 1,000        | 22.80 ms    | 0.39 ms    | 58.7×    |
| 5,000        | 229.76 ms   | 2.04 ms    | 112.4×   |
| 10,000       | 742.99 ms   | 2.59 ms    | 286.6×   |

#### Fraction Variations

| Fraction | statsmodels | fastLowess | Speedup |
|----------|-------------|------------|---------|
| 0.1      | 19.31 ms    | 0.36 ms    | 53.3×   |
| 0.2      | 21.20 ms    | 0.70 ms    | 30.2×   |
| 0.3      | 23.01 ms    | 0.51 ms    | 45.0×   |
| 0.5      | 25.82 ms    | 0.76 ms    | 33.9×   |
| 0.67     | 29.25 ms    | 0.63 ms    | 46.2×   |
| 0.8      | 31.40 ms    | 0.98 ms    | 32.1×   |

#### Robustness Iterations

| Iterations | statsmodels | fastLowess | Speedup |
|------------|-------------|------------|---------|
| 0          | 5.91 ms     | 0.16 ms    | 37.6×   |
| 1          | 11.65 ms    | 0.27 ms    | 43.2×   |
| 2          | 17.50 ms    | 0.58 ms    | 30.2×   |
| 3          | 22.86 ms    | 0.56 ms    | 41.1×   |
| 5          | 34.19 ms    | 0.88 ms    | 38.9×   |
| 10         | 62.81 ms    | 1.26 ms    | 49.8×   |

#### Pathological Cases

| Case             | statsmodels | fastLowess | Speedup  |
|------------------|-------------|------------|----------|
| clustered_x      | 19.36 ms    | 0.16 ms    | 124.0×   |
| constant_y       | 17.27 ms    | 0.41 ms    | 42.6×    |
| extreme_outliers | 34.94 ms    | 0.77 ms    | 45.4×    |
| high_noise       | 28.84 ms    | 0.46 ms    | 62.6×    |

#### Realistic Scenarios

| Scenario             | statsmodels | fastLowess | Speedup |
|----------------------|-------------|------------|---------|
| financial_timeseries | 14.73 ms    | 0.20 ms    | 72.0×   |
| scientific_data      | 21.52 ms    | 0.62 ms    | 34.5×   |
| genomic_methylation  | 21.45 ms    | 2.36 ms    | 9.1×    |

#### Delta Parameter

| Delta Config | statsmodels | fastLowess | Speedup |
|--------------|-------------|------------|---------|
| delta_none   | 171.76 ms   | 16.38 ms   | 10.5×   |
| delta_auto   | 3.88 ms     | 0.64 ms    | 6.1×    |
| delta_small  | 19.56 ms    | 2.21 ms    | 8.8×    |
| delta_large  | 2.07 ms     | 0.41 ms    | 5.1×    |

## Conclusion

`fastLowess` provides **5-287× speedup** over statsmodels for parallel LOWESS smoothing:

- ✅ **Best case**: 287× faster (large datasets with 10,000 points)
- ✅ **Typical case**: 30-60× faster (most workloads)
- ✅ **Delta optimization**: Properly implemented (5-10× speedup)
- ✅ **No regressions**: fastLowess is faster in all tested scenarios
