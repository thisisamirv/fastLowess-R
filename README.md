# fastLowess-R Benchmarking & Validation

This workspace is dedicated to validating the correctness and benchmarking the performance of the [fastLowess-R](https://github.com/thisisamirv/fastLowess-R) package against the reference R implementation (`stats::lowess`).

It ensures the Rust-backed implementation delivers significant performance wins while maintaining strict numerical parity.

## Structure

- `benchmarks/`: Performance benchmarking suite (fastLowess vs Base R).
- `validation/`: Correctness validation suite (fastLowess vs Statsmodels).

## How to Run Benchmarks

Benchmarks measure execution time across various scenarios (scalability, fraction, iterations, pathology).

### 1. Run FastLowess Benchmarks

```bash
Rscript benchmarks/fastLowess/benchmark.R
```

*Output: `benchmarks/output/fastLowess_benchmark.json`*

### 2. Run Base R Benchmarks

```bash
Rscript benchmarks/base_R/benchmark.R
```

*Output: `benchmarks/output/base_R_benchmark.json`*

### 3. Compare Results

Generate a comparison report showing speedups and regressions.

```bash
cd benchmarks
python3 compare_benchmark.py
```

*See `benchmarks/INTERPRETATION.md` for detailed results.*

## How to Run Validation

Validation ensures `fastLowess-R` produces results identical to `statsmodels`.

### 1. Run FastLowess Validation

```bash
Rscript validation/fastLowess/validate.R
```

### 2. Run Python Reference Validation

```bash
python3 validation/statsmodels/validate.py
```

### 3. Compare Results

```bash
cd validation
python3 compare_validation.py
```

## Requirements

- **R**: 4.3.1+ with `bench` and `jsonlite` installed.
- **Python**: 3.x with `numpy`, `scipy`, `statsmodels` (for cross-language validation).
- **FastLowess**: Installed via `devtools::install_github("thisisamirv/fastLowess-R")`.
