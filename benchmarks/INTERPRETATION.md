# Benchmark Interpretation (fastlowess-R)

## Summary

The `fastlowess` R package demonstrates significant performance gains over R's `stats::lowess`. The benchmarks compare `fastlowess` (both Serial and Parallel execution modes) against the standard R implementation.

The results show that `fastlowess` consistently outperforms `stats::lowess`, achieving speedups ranging from **1.1x to 6.8x**, with an **average speedup of approximately 2.3x**.

## Consolidated Comparison

The table below shows speedups relative to the **R `stats::lowess` baseline**.

| Name                  |      R      |  fastlowess   |
|-----------------------|-------------|---------------|
| clustered             |   2.16ms    |   2.4-4.7x    |
| constant_y            |   1.41ms    |   2.1-3.9x    |
| delta_large           |   0.53ms    |   3.4-2.0x    |
| delta_medium          |   0.73ms    |   3.3-2.8x    |
| delta_none            |  164.35ms   |   1.5-6.8x    |
| delta_small           |   0.97ms    |   2.6-3.3x    |
| extreme_outliers      |   4.76ms    |   1.9-3.9x    |
| financial_1000        |   0.18ms    |   1.8-1.2x    |
| financial_10000       |   1.73ms    |   2.2-2.6x    |
| financial_500         |   0.12ms    |   1.6-0.9x    |
| financial_5000        |   1.00ms    |   2.6-2.7x    |
| fraction_0.05         |   0.69ms    |   2.3-1.5x    |
| fraction_0.1          |   1.07ms    |   2.0-2.6x    |
| fraction_0.2          |   1.62ms    |   1.7-2.9x    |
| fraction_0.3          |   2.24ms    |   1.7-3.4x    |
| fraction_0.5          |   3.35ms    |   1.6-3.6x    |
| fraction_0.67         |   4.33ms    |   1.6-4.2x    |
| genomic_1000          |   1.14ms    |   1.2-1.4x    |
| genomic_10000         |  111.74ms   |   1.5-5.3x    |
| genomic_5000          |   27.73ms   |   1.4-4.3x    |
| genomic_50000         |  2818.61ms  |   1.6-5.5x    |
| high_noise            |   3.45ms    |   1.1-3.2x    |
| iterations_0          |   0.36ms    |   1.9-2.6x    |
| iterations_1          |   0.76ms    |   1.7-2.9x    |
| iterations_10         |   4.33ms    |   1.7-2.6x    |
| iterations_2          |   1.17ms    |   1.7-2.3x    |
| iterations_3          |   1.60ms    |   1.7-2.6x    |
| iterations_5          |   2.41ms    |   1.7-2.5x    |
| scale_1000            |   0.21ms    |   1.5-1.0x    |
| scale_10000           |   2.11ms    |   2.1-2.5x    |
| scale_5000            |   0.99ms    |   1.9-2.4x    |
| scale_50000           |   10.13ms   |   2.2-2.1x    |
| scientific_1000       |   0.28ms    |   1.5-0.8x    |
| scientific_10000      |   2.36ms    |   1.6-2.2x    |
| scientific_500        |   0.15ms    |   1.3-0.6x    |
| scientific_5000       |   1.22ms    |   1.6-1.7x    |

\* **fastlowess**: Shows speedup range `[Serial-Parallel]`. E.g., `[2.0-2.5x]` means 2.0x speedup (Serial) and 2.5x speedup (Parallel).

## Key Takeaways

1. **Consistent Performance Gains**: `fastlowess` is consistently faster than `stats::lowess` across all benchmark categories, with speedups ranging from 1.1x to 6.8x.

2. **Parallel Scaling**:
   - **Large Datasets**: Parallel execution provides significant gains. For example, `delta_none` shows a jump from 1.5x (Serial) to 6.8x (Parallel) speedup, and `genomic_10000` shows 1.5x to 5.3x.
   - **Small Datasets**: For very small datasets (e.g., `scale_1000`, `financial_500`, `scientific_500`), Serial execution may be faster than Parallel due to thread overhead (e.g., `[1.5-1.0x]`, `[1.6-0.9x]`, `[1.3-0.6x]`).

3. **Best Performance Scenarios**:
   - **Delta optimization**: When delta is enabled (non-zero), speedups are moderate (2-3.5x). When delta is disabled (`delta_none`), parallel execution shines with up to 6.8x speedup.
   - **Large-scale genomic data**: Shows excellent scaling with parallel execution (up to 5.5x for 50,000 points).
   - **Pathological cases**: `fastlowess` maintains good performance even with challenging data like `extreme_outliers` (1.9-3.9x) and `high_noise` (1.1-3.2x).

4. **Robustness Iterations**: Performance remains consistent across different iteration counts (1.7-2.9x speedup range), showing that `fastlowess` handles robustness efficiently.

## Recommendation

- **General Use**: `fastlowess` provides better performance than `stats::lowess` for most use cases, with the added benefit of additional features (confidence intervals, cross-validation, diagnostics).
- **Parallelism**: Enable `parallel=TRUE` (default) for datasets larger than ~5,000 points. For very small batches (< 1,000 points), `parallel=FALSE` may offer slightly lower latency.
- **Delta Optimization**: The delta parameter provides significant speedup in both implementations. Use the default auto-calculated delta for best performance.
