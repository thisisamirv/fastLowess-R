#!/usr/bin/Rscript

# R fastLowess benchmark runner with JSON output for comparison with statsmodels.
# This is a standalone benchmark program that outputs results in JSON format.

suppressPackageStartupMessages({
    library(jsonlite)
    library(fastLowess)
})

# ============================================================================
# Constants
# ============================================================================

WARMUP_ITERATIONS <- 3

# ============================================================================
# Data Structures
# ============================================================================

create_benchmark_result <- function(name, size, iterations) {
    list(
        name = name,
        size = size,
        iterations = iterations,
        mean_time_ms = 0.0,
        std_time_ms = 0.0,
        median_time_ms = 0.0,
        min_time_ms = 0.0,
        max_time_ms = 0.0
    )
}

compute_stats <- function(result, times) {
    if (length(times) == 0) return(result)
    
    # Convert to milliseconds
    times_ms <- times * 1000.0
    
    result$mean_time_ms <- mean(times_ms)
    result$std_time_ms <- sd(times_ms)
    result$median_time_ms <- median(times_ms)
    result$min_time_ms <- min(times_ms)
    result$max_time_ms <- max(times_ms)
    
    return(result)
}

# ============================================================================
# Data Generation
# ============================================================================

generate_data <- function(size) {
    x <- seq(0, 10.0, length.out = size)
    i <- 0:(size-1)
    x <- i * 10.0 / size
    
    y <- sin(x) + sin(sin((i * 7.3) * 0.5)) * 0.2
    
    list(x = x, y = y)
}

generate_data_with_outliers <- function(size) {
    data <- generate_data(size)
    x <- data$x
    
    y <- sin(x)
    
    # Add outliers (5% of points)
    n_outliers <- max(floor(size / 20), 1)
    for (k in 0:(n_outliers-1)) {
        idx <- floor((k * size) / n_outliers) + 1
        if (k %% 2 == 0) {
            y[idx] <- y[idx] + 3.0
        } else {
            y[idx] <- y[idx] - 3.0
        }
    }
    
    list(x = x, y = y)
}

# ============================================================================
# Benchmark Functions
# ============================================================================

benchmark_basic_smoothing <- function(sizes, iterations) {
    results <- list()
    
    for (size in sizes) {
        cat(sprintf("Benchmarking basic smoothing with size=%d...\n", size))
        result <- create_benchmark_result(paste0("basic_smoothing_", size), size, iterations)
        
        data <- generate_data(size)
        x <- data$x
        y <- data$y
        
        times <- numeric(iterations)
        
        # Warmup
        for (i in 1:WARMUP_ITERATIONS) {
            invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=3, parallel=TRUE))
        }
        
        # Benchmark
        for (i in 1:iterations) {
            start_time <- Sys.time()
            invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=3, parallel=TRUE))
            end_time <- Sys.time()
            times[i] <- as.numeric(end_time - start_time)
        }
        
        result <- compute_stats(result, times)
        cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
        results[[length(results) + 1]] <- result
    }
    
    return(results)
}

benchmark_fraction_variations <- function(size, iterations) {
    results <- list()
    fractions <- c(0.1, 0.2, 0.3, 0.5, 0.67, 0.8)
    
    data <- generate_data(size)
    x <- data$x
    y <- data$y
    
    for (frac in fractions) {
        cat(sprintf("Benchmarking fraction=%.2f...\n", frac))
        result <- create_benchmark_result(paste0("fraction_", frac), size, iterations)
        
        times <- numeric(iterations)
        
        # Warmup
        for (i in 1:WARMUP_ITERATIONS) {
            invisible(fastLowess::smooth(x, y, fraction=frac, iterations=3, parallel=TRUE))
        }
        
        # Benchmark
        for (i in 1:iterations) {
            start_time <- Sys.time()
            invisible(fastLowess::smooth(x, y, fraction=frac, iterations=3, parallel=TRUE))
            end_time <- Sys.time()
            times[i] <- as.numeric(end_time - start_time)
        }
        
        result <- compute_stats(result, times)
        cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
        results[[length(results) + 1]] <- result
    }
    
    return(results)
}

benchmark_robustness_iterations <- function(size, iterations) {
    results <- list()
    niter_values <- c(0, 1, 2, 3, 5, 10)
    
    data <- generate_data_with_outliers(size)
    x <- data$x
    y <- data$y
    
    for (niter in niter_values) {
        cat(sprintf("Benchmarking robustness iterations=%d...\n", niter))
        result <- create_benchmark_result(paste0("iterations_", niter), size, iterations)
        
        times <- numeric(iterations)
        
        # Warmup
        for (i in 1:WARMUP_ITERATIONS) {
            invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=niter, parallel=TRUE))
        }
        
        # Benchmark
        for (i in 1:iterations) {
            start_time <- Sys.time()
            invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=niter, parallel=TRUE))
            end_time <- Sys.time()
            times[i] <- as.numeric(end_time - start_time)
        }
        
        result <- compute_stats(result, times)
        cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
        results[[length(results) + 1]] <- result
    }
    
    return(results)
}

benchmark_delta_parameter <- function(size, iterations) {
    results <- list()
    
    i <- 0:(size-1)
    x <- i * 0.1
    y <- sin(x)
    
    delta_configs <- list(
        list(name="delta_none", val=0.0),
        list(name="delta_auto", val=NULL),
        list(name="delta_small", val=1.0),
        list(name="delta_large", val=10.0)
    )
    
    for (config in delta_configs) {
        delta_val <- config$val
        name <- config$name
        
        cat(sprintf("Benchmarking %s (delta=%s)...\n", name, ifelse(is.null(delta_val), "NULL", as.character(delta_val))))
        result <- create_benchmark_result(name, size, iterations)
        
        times <- numeric(iterations)
        
        # Warmup
        for (i in 1:WARMUP_ITERATIONS) {
            if (is.null(delta_val)) {
                invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=2, parallel=TRUE))
            } else {
                invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=2, delta=delta_val, parallel=TRUE))
            }
        }
        
        # Benchmark
        for (i in 1:iterations) {
            start_time <- Sys.time()
            if (is.null(delta_val)) {
                invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=2, parallel=TRUE))
            } else {
                invisible(fastLowess::smooth(x, y, fraction=0.3, iterations=2, delta=delta_val, parallel=TRUE))
            }
            end_time <- Sys.time()
            times[i] <- as.numeric(end_time - start_time)
        }
        
        result <- compute_stats(result, times)
        cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
        results[[length(results) + 1]] <- result
    }
    
    return(results)
}

benchmark_pathological_cases <- function(size, iterations) {
    results <- list()
    
    # Clustered x values
    cat("Benchmarking clustered_x...\n")
    i <- 0:(size-1)
    x_clustered <- floor(i / 100) + (i %% 100) * 1e-6
    y_clustered <- sin(x_clustered)
    
    result <- create_benchmark_result("clustered_x", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_clustered, y_clustered, fraction=0.5, iterations=2, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_clustered, y_clustered, fraction=0.5, iterations=2, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    # Extreme outliers
    cat("Benchmarking extreme_outliers...\n")
    i <- 0:(size-1)
    x_normal <- i * 10.0 / size
    y_outliers <- sin(x_normal)
    
    # Python: for i in range(0, size, 50): y[i] += ...
    # R: 1-based indexing so loop needs care. Python range(0, size, 50) means 0, 50, 100...
    indices <- seq(0, size-1, by=50)
    for (idx in indices) {
        r_idx <- idx + 1
        if (idx %% 100 == 0) {
            y_outliers[r_idx] <- y_outliers[r_idx] + 100.0
        } else {
            y_outliers[r_idx] <- y_outliers[r_idx] - 100.0
        }
    }
    
    result <- create_benchmark_result("extreme_outliers", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_normal, y_outliers, fraction=0.3, iterations=5, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_normal, y_outliers, fraction=0.3, iterations=5, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    # Constant y values
    cat("Benchmarking constant_y...\n")
    y_constant <- rep(5.0, size)
    
    result <- create_benchmark_result("constant_y", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_normal, y_constant, fraction=0.3, iterations=2, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_normal, y_constant, fraction=0.3, iterations=2, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    # High noise
    cat("Benchmarking high_noise...\n")
    i <- 0:(size-1)
    y_noisy <- sin(x_normal / 10.0) * 0.1 + sin(sin((i * 7.3) * 0.5)) * 2.0
    
    result <- create_benchmark_result("high_noise", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_normal, y_noisy, fraction=0.6, iterations=3, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_normal, y_noisy, fraction=0.6, iterations=3, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    return(results)
}

benchmark_realistic_scenarios <- function(iterations) {
    results <- list()
    size <- 1000
    
    # Financial time series
    cat("Benchmarking financial_timeseries...\n")
    i <- 0:(size-1)
    x <- as.numeric(i)
    y <- x * 0.01 + sin(x / 50.0) * 0.5 + sin(sin((i * 7.3) * 0.5)) * 0.3
    
    result <- create_benchmark_result("financial_timeseries", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x, y, fraction=0.1, iterations=2, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x, y, fraction=0.1, iterations=2, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    # Scientific data
    cat("Benchmarking scientific_data...\n")
    i <- 0:(size-1)
    x_sci <- i * 0.01
    y_sci <- exp(x_sci * 2.0 * pi) * cos(x_sci * 10.0) + sin(sin((i * 13.7) * 0.3)) * 0.1
    
    result <- create_benchmark_result("scientific_data", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_sci, y_sci, fraction=0.2, iterations=3, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_sci, y_sci, fraction=0.2, iterations=3, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    # Genomic methylation
    cat("Benchmarking genomic_methylation...\n")
    i <- 0:(size-1)
    x_genomic <- as.numeric(i * 1000)
    raw_y <- 0.5 + sin(x_genomic / 5000.0) * 0.2 + sin(sin((i * 17.3) * 0.3)) * 0.15
    y_genomic <- pmax(0.0, pmin(1.0, raw_y))
    
    result <- create_benchmark_result("genomic_methylation", size, iterations)
    times <- numeric(iterations)
    
    for (i in 1:WARMUP_ITERATIONS) {
        invisible(fastLowess::smooth(x_genomic, y_genomic, fraction=0.2, iterations=3, delta=100.0, parallel=TRUE))
    }
    for (i in 1:iterations) {
        start <- Sys.time()
        invisible(fastLowess::smooth(x_genomic, y_genomic, fraction=0.2, iterations=3, delta=100.0, parallel=TRUE))
        times[i] <- as.numeric(Sys.time() - start)
    }
    result <- compute_stats(result, times)
    cat(sprintf("  Mean: %.2f ms ± %.2f ms\n", result$mean_time_ms, result$std_time_ms))
    results[[length(results) + 1]] <- result
    
    return(results)
}

# ============================================================================
# Main
# ============================================================================

main <- function() {
    cat(paste0(rep("=", 80), collapse=""), "\n")
    cat("R FASTLOWESS BENCHMARK SUITE\n")
    cat(paste0(rep("=", 80), collapse=""), "\n\n")
    
    all_results <- list()
    
    # Core benchmarks
    cat("\n", paste0(rep("=", 80), collapse=""), "\n")
    cat("CORE BENCHMARKS\n")
    cat(paste0(rep("=", 80), collapse=""), "\n\n")
    
    all_results[["basic_smoothing"]] <- benchmark_basic_smoothing(c(100, 500, 1000, 5000, 10000), 10)
    
    all_results[["fraction_variations"]] <- benchmark_fraction_variations(1000, 10)
    
    all_results[["robustness_iterations"]] <- benchmark_robustness_iterations(1000, 10)
    
    all_results[["delta_parameter"]] <- benchmark_delta_parameter(5000, 10)
    
    # Stress tests
    cat("\n", paste0(rep("=", 80), collapse=""), "\n")
    cat("STRESS TESTS\n")
    cat(paste0(rep("=", 80), collapse=""), "\n\n")
    
    all_results[["pathological_cases"]] <- benchmark_pathological_cases(1000, 10)
    
    # Application scenarios
    cat("\n", paste0(rep("=", 80), collapse=""), "\n")
    cat("APPLICATION SCENARIOS\n")
    cat(paste0(rep("=", 80), collapse=""), "\n\n")
    
    all_results[["realistic_scenarios"]] <- benchmark_realistic_scenarios(10)
    
    # Save results
    json_str <- toJSON(all_results, auto_unbox = TRUE, pretty = TRUE)
    out_dir <- "benchmarks/output"
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
    }
    
    out_path <- file.path(out_dir, "fastLowess_benchmark.json")
    write(json_str, file = out_path)
    
    cat("\n", paste0(rep("=", 80), collapse=""), "\n")
    cat(sprintf("Results saved to %s\n", out_path))
    cat(paste0(rep("=", 80), collapse=""), "\n")
}

main()
