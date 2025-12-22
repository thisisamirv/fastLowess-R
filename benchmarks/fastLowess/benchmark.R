#!/usr/bin/Rscript
# Industry-level LOWESS benchmarks for fastlowess.
# Aligned with the fastLowess-py and statsmodels benchmarks.
# Results are written to benchmarks/output/fastLowess_benchmark.json.
#
# Run with: Rscript benchmark.R

suppressPackageStartupMessages({
  library(fastlowess)
  library(bench)
  library(jsonlite)
})

# ============================================================================
# Data Generation (Aligned with Python implementation)
# ============================================================================

generate_sine_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  y <- sin(x) + rnorm(size, 0, 0.2)
  list(x = x, y = y)
}

generate_outlier_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  y <- sin(x) + rnorm(size, 0, 0.2)

  # Add 5% outliers
  n_outliers <- floor(size / 20)
  if (n_outliers > 0) {
    indices <- sample(1:size, n_outliers)
    y[indices] <- y[indices] + runif(n_outliers, -5, 5)
  }
  list(x = x, y = y)
}

generate_financial_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- 0:(size - 1)
  y <- numeric(size)
  y[1] <- 100.0
  for (i in 2:size) {
    ret <- rnorm(1, 0.0005, 0.02)
    y[i] <- y[i - 1] * (1 + ret)
  }
  list(x = as.numeric(x), y = y)
}

generate_scientific_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  signal <- exp(-x * 0.3) * cos(x * 2 * pi)
  noise <- rnorm(size, 0, 0.05)
  list(x = x, y = signal + noise)
}

generate_genomic_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- (0:(size - 1)) * 1000.0
  base <- 0.5 + sin(x / 50000.0) * 0.3
  noise <- rnorm(size, 0, 0.1)
  y <- pmax(0.0, pmin(1.0, base + noise))
  list(x = x, y = y)
}

generate_clustered_data <- function(size, seed = 42) {
  set.seed(seed)
  i <- 0:(size - 1)
  x <- (i %/% 100) + (i %% 100) * 1e-6
  y <- sin(x) + rnorm(size, 0, 0.1)
  list(x = x, y = y)
}

generate_high_noise_data <- function(size, seed = 42) {
  set.seed(seed)
  x <- seq(0, 10, length.out = size)
  signal <- sin(x) * 0.5
  noise <- rnorm(size, 0, 2.0)
  list(x = x, y = signal + noise)
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("FASTLOWESS R BENCHMARK SUITE (Aligned with Python)\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  iterations <- 10
  all_results <- list()

  # 1. Scalability
  cat("\nSCALABILITY\n")
  sizes <- c(1000, 5000, 10000, 50000, 100000)
  scale_results <- list()
  for (size in sizes) {
    data <- generate_sine_data(size)
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.1, iterations = 3L),
      iterations = iterations, check = FALSE
    )
    scale_results[[length(scale_results) + 1]] <- list(
      name = paste0("scale_", size), size = size, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  scale_%d: %.2f ms\n", size, as.numeric(res$median) * 1000))
  }
  all_results$scalability <- scale_results

  # 2. Fraction
  cat("\nFRACTION\n")
  data <- generate_sine_data(5000)
  fractions <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.67)
  frac_results <- list()
  for (f in fractions) {
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = f, iterations = 3L),
      iterations = iterations, check = FALSE
    )
    frac_results[[length(frac_results) + 1]] <- list(
      name = paste0("fraction_", f), size = 5000, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  fraction_%.2f: %.2f ms\n", f, as.numeric(res$median) * 1000))
  }
  all_results$fraction <- frac_results

  # 3. Iterations
  cat("\nITERATIONS\n")
  data <- generate_outlier_data(5000)
  iters <- c(0, 1, 2, 3, 5, 10)
  iter_results <- list()
  for (it in iters) {
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.2, iterations = as.integer(it)),
      iterations = iterations, check = FALSE
    )
    iter_results[[length(iter_results) + 1]] <- list(
      name = paste0("iterations_", it), size = 5000, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  iterations_%d: %.2f ms\n", it, as.numeric(res$median) * 1000))
  }
  all_results$iterations <- iter_results

  # 4. Delta
  cat("\nDELTA\n")
  data <- generate_sine_data(10000)
  deltas <- list(list("delta_none", 0.0), list("delta_small", 0.5), list("delta_medium", 2.0), list("delta_large", 10.0))
  delta_results <- list()
  for (d in deltas) {
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.2, iterations = 2L, delta = d[[2]]),
      iterations = iterations, check = FALSE
    )
    delta_results[[length(delta_results) + 1]] <- list(
      name = d[[1]], size = 10000, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  %s: %.2f ms\n", d[[1]], as.numeric(res$median) * 1000))
  }
  all_results$delta <- delta_results

  # 5. Financial
  cat("\nFINANCIAL\n")
  fin_results <- list()
  for (size in c(500, 1000, 5000, 10000)) {
    data <- generate_financial_data(size)
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.1, iterations = 2L),
      iterations = iterations, check = FALSE
    )
    fin_results[[length(fin_results) + 1]] <- list(
      name = paste0("financial_", size), size = size, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  financial_%d: %.2f ms\n", size, as.numeric(res$median) * 1000))
  }
  all_results$financial <- fin_results

  # 6. Scientific
  cat("\nSCIENTIFIC\n")
  sci_results <- list()
  for (size in c(500, 1000, 5000, 10000)) {
    data <- generate_scientific_data(size)
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.15, iterations = 3L),
      iterations = iterations, check = FALSE
    )
    sci_results[[length(sci_results) + 1]] <- list(
      name = paste0("scientific_", size), size = size, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  scientific_%d: %.2f ms\n", size, as.numeric(res$median) * 1000))
  }
  all_results$scientific <- sci_results

  # 7. Genomic
  cat("\nGENOMIC\n")
  gen_results <- list()
  for (size in c(1000, 5000, 10000, 50000)) {
    data <- generate_genomic_data(size)
    res <- bench::mark(
      fastlowess::smooth(data$x, data$y, fraction = 0.1, iterations = 3L, delta = 100.0),
      iterations = iterations, check = FALSE
    )
    gen_results[[length(gen_results) + 1]] <- list(
      name = paste0("genomic_", size), size = size, iterations = iterations,
      mean_time_ms = as.numeric(res$median) * 1000,
      median_time_ms = as.numeric(res$median) * 1000,
      min_time_ms = as.numeric(res$min) * 1000,
      max_time_ms = as.numeric(max(res$time[[1]])) * 1000
    )
    cat(sprintf("  genomic_%d: %.2f ms\n", size, as.numeric(res$median) * 1000))
  }
  all_results$genomic <- gen_results

  # 8. Pathological
  cat("\nPATHOLOGICAL\n")
  path_results <- list()
  size <- 5000

  # Clustered
  data <- generate_clustered_data(size)
  res <- bench::mark(fastlowess::smooth(data$x, data$y, fraction = 0.3, iterations = 2L), iterations = iterations, check = FALSE)
  path_results[[1]] <- list(name = "clustered", size = size, iterations = iterations, mean_time_ms = as.numeric(res$median) * 1000, median_time_ms = as.numeric(res$median) * 1000, min_time_ms = as.numeric(res$min) * 1000, max_time_ms = as.numeric(max(res$time[[1]])) * 1000)

  # High Noise
  data <- generate_high_noise_data(size)
  res <- bench::mark(fastlowess::smooth(data$x, data$y, fraction = 0.5, iterations = 5L), iterations = iterations, check = FALSE)
  path_results[[2]] <- list(name = "high_noise", size = size, iterations = iterations, mean_time_ms = as.numeric(res$median) * 1000, median_time_ms = as.numeric(res$median) * 1000, min_time_ms = as.numeric(res$min) * 1000, max_time_ms = as.numeric(max(res$time[[1]])) * 1000)

  # Extreme Outliers
  data <- generate_outlier_data(size)
  res <- bench::mark(fastlowess::smooth(data$x, data$y, fraction = 0.2, iterations = 10L), iterations = iterations, check = FALSE)
  path_results[[3]] <- list(name = "extreme_outliers", size = size, iterations = iterations, mean_time_ms = as.numeric(res$median) * 1000, median_time_ms = as.numeric(res$median) * 1000, min_time_ms = as.numeric(res$min) * 1000, max_time_ms = as.numeric(max(res$time[[1]])) * 1000)

  # Constant Y
  data <- list(x = as.numeric(0:(size - 1)), y = rep(5.0, size))
  res <- bench::mark(fastlowess::smooth(data$x, data$y, fraction = 0.2, iterations = 2L), iterations = iterations, check = FALSE)
  path_results[[4]] <- list(name = "constant_y", size = size, iterations = iterations, mean_time_ms = as.numeric(res$median) * 1000, median_time_ms = as.numeric(res$median) * 1000, min_time_ms = as.numeric(res$min) * 1000, max_time_ms = as.numeric(max(res$time[[1]])) * 1000)

  all_results$pathological <- path_results

  # Save to benchmarks/output directory
  # Find centralized benchmarks/output directory
  if (dir.exists("benchmarks/output")) {
    final_output_dir <- "benchmarks/output"
  } else if (file.exists("../compare_benchmark.py") && dir.exists("../output")) {
    final_output_dir <- "../output"
  } else {
    final_output_dir <- "output"
  }

  if (!dir.exists(final_output_dir)) dir.create(final_output_dir, recursive = TRUE)
  out_path <- file.path(final_output_dir, "fastLowess_benchmark.json")
  write_json(all_results, out_path, auto_unbox = TRUE, pretty = TRUE)

  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat(sprintf("Results saved to %s\n", out_path))
  cat(paste(rep("=", 80), collapse = ""), "\n")
}

main()
