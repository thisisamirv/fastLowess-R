# Industry-level LOWESS benchmarks with JSON output for comparison
#
# Benchmarks are aligned with the Python and Rust criterion benchmarks.
# Results are written to benchmarks/output/fastlowess_benchmark.json.
#
# Run with: Rscript benchmark.R

library(jsonlite)
library(fastlowess)

# ============================================================================
# Data Generation (Aligned with Python/Rust)
# ============================================================================

generate_sine_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10.0, length.out = size)
  y <- sin(x) + rnorm(size, mean = 0.0, sd = 0.2)
  list(x = x, y = y)
}

generate_outlier_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10.0, length.out = size)
  y <- sin(x) + rnorm(size, mean = 0.0, sd = 0.2)

  n_outliers <- as.integer(size / 20)
  outlier_indices <- sample(1:size, n_outliers)
  y[outlier_indices] <- y[outlier_indices] + runif(n_outliers, min = -5.0, max = 5.0)

  list(x = x, y = y)
}

generate_financial_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, size - 1, by = 1)
  y <- numeric(size)
  y[1] <- 100.0
  returns <- rnorm(size - 1, mean = 0.0005, sd = 0.02)
  y[2:size] <- 100.0 * cumprod(1.0 + returns)
  list(x = x, y = y)
}

generate_scientific_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, size * 0.01, length.out = size)
  signal <- exp(-x * 0.3) * cos(x * 2.0 * pi)
  y <- signal + rnorm(size, mean = 0.0, sd = 0.05)
  list(x = x, y = y)
}

generate_genomic_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, size - 1, by = 1) * 1000.0
  base <- 0.5 + sin(x / 50000.0) * 0.3
  noise <- rnorm(size, mean = 0.0, sd = 0.1)
  y <- pmin(pmax(base + noise, 0.0), 1.0)
  list(x = x, y = y)
}

generate_clustered_data <- function(size, seed = 42) {
  set.seed(seed)
  indices <- seq(0, size - 1, by = 1)
  x <- as.numeric(indices %/% 100) + as.numeric(indices %% 100) * 1e-6
  y <- sin(x) + rnorm(size, mean = 0.0, sd = 0.1)
  list(x = x, y = y)
}

generate_high_noise_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10.0, length.out = size)
  y <- sin(x) * 0.5 + rnorm(size, mean = 0.0, sd = 2.0)
  list(x = x, y = y)
}

# ============================================================================
# Benchmark Execution
# ============================================================================

run_benchmark <- function(name, size, func, iterations = 10, warmup = 2) {
  # Warmup runs
  for (i in 1:warmup) {
    func()
  }

  # Timed runs
  times <- numeric(iterations)
  for (i in 1:iterations) {
    start <- Sys.time()
    func()
    elapsed <- Sys.time() - start
    times[i] <- as.numeric(elapsed) * 1000 # Convert to ms
  }

  list(
    name = name,
    size = size,
    iterations = iterations,
    mean_time_ms = mean(times),
    std_time_ms = sd(times),
    median_time_ms = median(times),
    min_time_ms = min(times),
    max_time_ms = max(times)
  )
}

# ============================================================================
# Benchmark Categories
# ============================================================================

benchmark_scalability <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("SCALABILITY (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  sizes <- c(1000, 5000, 10000, 50000)

  for (size in sizes) {
    data <- generate_sine_data(size, seed = 42)

    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.1,
        iterations = 3L,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("scale_", size), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  scale_%d: %.4f ms ± %.4f ms\n",
      size, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_fraction <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("FRACTION (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  size <- 5000
  fractions <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.67)
  data <- generate_sine_data(size, seed = 42)

  for (frac in fractions) {
    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = frac,
        iterations = 3L,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("fraction_", frac), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  fraction_%.2f: %.4f ms ± %.4f ms\n",
      frac, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_iterations <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("ITERATIONS (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  size <- 5000
  iter_values <- c(0, 1, 2, 3, 5, 10)
  data <- generate_outlier_data(size, seed = 42)

  for (it in iter_values) {
    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.2,
        iterations = as.integer(it),
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("iterations_", it), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  iterations_%d: %.4f ms ± %.4f ms\n",
      it, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_delta <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("DELTA (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  size <- 10000
  data <- generate_sine_data(size, seed = 42)

  delta_configs <- list(
    list(name = "delta_none", delta = 0.0),
    list(name = "delta_small", delta = 0.5),
    list(name = "delta_medium", delta = 2.0),
    list(name = "delta_large", delta = 10.0)
  )

  for (config in delta_configs) {
    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.2,
        iterations = 2L,
        delta = config$delta,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(config$name, size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  %s: %.4f ms ± %.4f ms\n",
      config$name, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_financial <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("FINANCIAL (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  sizes <- c(500, 1000, 5000, 10000)

  for (size in sizes) {
    data <- generate_financial_data(size, seed = 42)

    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.1,
        iterations = 2L,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("financial_", size), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  financial_%d: %.4f ms ± %.4f ms\n",
      size, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_scientific <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("SCIENTIFIC (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  sizes <- c(500, 1000, 5000, 10000)

  for (size in sizes) {
    data <- generate_scientific_data(size, seed = 42)

    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.15,
        iterations = 3L,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("scientific_", size), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  scientific_%d: %.4f ms ± %.4f ms\n",
      size, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_genomic <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("GENOMIC (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  sizes <- c(1000, 5000, 10000, 50000)

  for (size in sizes) {
    data <- generate_genomic_data(size, seed = 42)

    run_func <- function() {
      smooth(
        data$x, data$y,
        fraction = 0.1,
        iterations = 3L,
        delta = 100.0,
        scaling_method = "mar",
        boundary_policy = "noboundary",
        parallel = parallel
      )
    }

    result <- run_benchmark(paste0("genomic_", size), size, run_func, iterations)
    results[[length(results) + 1]] <- result
    cat(sprintf(
      "  genomic_%d: %.4f ms ± %.4f ms\n",
      size, result$mean_time_ms, result$std_time_ms
    ))
  }

  results
}

benchmark_pathological <- function(iterations = 10, parallel = TRUE) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("PATHOLOGICAL (Parallel=", parallel, ")\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  results <- list()
  size <- 5000

  # Clustered
  data_clustered <- generate_clustered_data(size, seed = 42)
  result <- run_benchmark(
    "clustered", size,
    function() {
      smooth(
        data_clustered$x, data_clustered$y,
        fraction = 0.3, iterations = 2L,
        scaling_method = "mar", boundary_policy = "noboundary", parallel = parallel
      )
    },
    iterations
  )
  results[[length(results) + 1]] <- result
  cat(sprintf(
    "  clustered: %.4f ms ± %.4f ms\n",
    result$mean_time_ms, result$std_time_ms
  ))

  # High noise
  data_noisy <- generate_high_noise_data(size, seed = 42)
  result <- run_benchmark(
    "high_noise", size,
    function() {
      smooth(
        data_noisy$x, data_noisy$y,
        fraction = 0.5, iterations = 5L,
        scaling_method = "mar", boundary_policy = "noboundary", parallel = parallel
      )
    },
    iterations
  )
  results[[length(results) + 1]] <- result
  cat(sprintf(
    "  high_noise: %.4f ms ± %.4f ms\n",
    result$mean_time_ms, result$std_time_ms
  ))

  # Extreme outliers
  data_outlier <- generate_outlier_data(size, seed = 42)
  result <- run_benchmark(
    "extreme_outliers", size,
    function() {
      smooth(
        data_outlier$x, data_outlier$y,
        fraction = 0.2, iterations = 10L,
        scaling_method = "mar", boundary_policy = "noboundary", parallel = parallel
      )
    },
    iterations
  )
  results[[length(results) + 1]] <- result
  cat(sprintf(
    "  extreme_outliers: %.4f ms ± %.4f ms\n",
    result$mean_time_ms, result$std_time_ms
  ))

  # Constant y
  x_const <- seq(0, size - 1, by = 1)
  y_const <- rep(5.0, size)
  result <- run_benchmark(
    "constant_y", size,
    function() {
      smooth(
        x_const, y_const,
        fraction = 0.2, iterations = 2L,
        scaling_method = "mar", boundary_policy = "noboundary", parallel = parallel
      )
    },
    iterations
  )
  results[[length(results) + 1]] <- result
  cat(sprintf(
    "  constant_y: %.4f ms ± %.4f ms\n",
    result$mean_time_ms, result$std_time_ms
  ))

  results
}

# ============================================================================
# Main Entry Point
# ============================================================================

run_suite <- function(parallel, output_filename) {
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("FASTLOWESS BENCHMARK SUITE (Parallel=", parallel, ")\n", sep = "")
  cat("Output: ", output_filename, "\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")

  iterations <- 25 # Reduced from 50 to save time double running

  # Run all benchmark categories
  all_results <- list(
    scalability = benchmark_scalability(iterations, parallel),
    fraction = benchmark_fraction(iterations, parallel),
    iterations = benchmark_iterations(iterations, parallel),
    delta = benchmark_delta(iterations, parallel),
    financial = benchmark_financial(iterations, parallel),
    scientific = benchmark_scientific(iterations, parallel),
    genomic = benchmark_genomic(iterations, parallel),
    pathological = benchmark_pathological(iterations, parallel)
  )

  # Save to output directory
  # Get script directory (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    script_dir <- dirname(script_path)
  } else {
    script_dir <- getwd()
  }
  benchmarks_dir <- dirname(script_dir)
  out_dir <- file.path(benchmarks_dir, "output")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  out_path <- file.path(out_dir, output_filename)
  write_json(all_results, out_path, pretty = TRUE, auto_unbox = TRUE)

  cat("\n", rep("=", 80), "\n", sep = "")
  cat("Results saved to ", out_path, "\n", sep = "")
  cat(rep("=", 80), "\n", sep = "")
}

main <- function() {
  # Run Parallel (Standard)
  run_suite(parallel = TRUE, output_filename = "fastlowess_benchmark.json")

  # Run Serial
  run_suite(parallel = FALSE, output_filename = "fastlowess_benchmark_serial.json")
}

# Execute main
main()
