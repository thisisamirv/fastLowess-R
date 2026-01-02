# Industry-level LOWESS validation for R with JSON I/O
#
# Reads validation test cases from ../output/R/ and writes results to ../output/fastLowess/
# Matches the Python validation script behavior
#
# Run with: Rscript validate.R

library(jsonlite)
library(fastlowess)

process_file <- function(input_path, output_dir) {
  cat(sprintf("Processing %s\n", basename(input_path)))

  # Read JSON file
  data <- tryCatch(
    {
      fromJSON(input_path, simplifyVector = FALSE)
    },
    error = function(e) {
      cat(sprintf("Failed to read %s: %s\n", basename(input_path), e$message))
      return(NULL)
    }
  )

  if (is.null(data)) {
    return()
  }

  # Extract params and input data
  params <- data$params
  input_data <- data$input

  x <- as.numeric(unlist(input_data$x))
  y <- as.numeric(unlist(input_data$y))

  fraction <- params$fraction
  iterations <- as.integer(params$iterations)
  delta <- params$delta

  # Run smoothing with matching parameters
  tryCatch(
    {
      # Configure arguments matching Python/Rust validation
      result <- smooth(
        x, y,
        fraction = fraction,
        iterations = iterations,
        delta = if (!is.null(delta)) delta else NULL,
        scaling_method = "mar", # Matching Rust validate.rs
        boundary_policy = "noboundary", # Matching Rust validate.rs
        parallel = FALSE # Matching Rust validate.rs
      )

      # Update result in data structure
      if (is.null(data$result)) {
        data$result <- list()
      }
      data$result$fitted <- result$y

      # Write output with full precision (15 decimal places)
      output_path <- file.path(output_dir, basename(input_path))
      write_json(data, output_path, auto_unbox = TRUE, pretty = TRUE, digits = 15)
    },
    error = function(e) {
      cat(sprintf("Failed to process %s: %s\n", basename(input_path), e$message))
    }
  )
}

main <- function() {
  # Get script directory (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    script_dir <- dirname(script_path)
  } else {
    script_dir <- getwd()
  }

  # Set up paths
  validation_dir <- dirname(script_dir)
  input_dir <- file.path(validation_dir, "output", "r")
  output_dir <- file.path(validation_dir, "output", "fastlowess")

  if (!dir.exists(input_dir)) {
    cat(sprintf("Input directory %s does not exist. Run R/validate.R first.\n", input_dir))
    return()
  }

  # Create output directory
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # Process all JSON files
  json_files <- list.files(input_dir, pattern = "\\.json$", full.names = TRUE)
  json_files <- sort(json_files)

  for (file_path in json_files) {
    process_file(file_path, output_dir)
  }

  cat("\n============================================================================\n")
  cat(sprintf("Validation complete. Results saved to %s\n", output_dir))
  cat("============================================================================\n")
}

# Execute main
main()
