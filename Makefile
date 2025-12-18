# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc
	@echo "All checks completed successfully!"

# Run all checks including R linting
check-all: check lint-r
	@echo "All checks (Rust + R) completed successfully!"

# --- Formatting ---
fmt: fmt-rust
	@echo "Formatting check complete!"

fmt-rust:
	@echo "Checking Rust code formatting..."
	@cargo fmt --all -- --check

fmt-fix: fmt-fix-rust
	@echo "Formatting complete!"

fmt-fix-rust:
	@echo "Formatting Rust code..."
	@cargo fmt --all

# --- Linter ---
clippy: clippy-default clippy-serial

clippy-default:
	@echo "Running clippy (default / parallel)..."
	@cargo clippy --all-targets -- -D warnings

clippy-serial:
	@echo "Running clippy (serial / no parallel)..."
	@cargo clippy --all-targets --no-default-features -- -D warnings
	@echo "Clippy check complete!"

lint-r:
	@echo "Linting R code with lintr..."
	@Rscript -e "lintr::lint_package()" || true
	@echo "R lint complete!"

# --- Build ---
build: build-default build-serial

build-default:
	@echo "Building crate (default / parallel)..."
	@cargo build

build-serial:
	@echo "Building crate (serial / no parallel)..."
	@cargo build --no-default-features
	@echo "Build complete!"

# --- R Package ---
install:
	@echo "Building and installing R package..."
	@R CMD build .
	@R CMD INSTALL fastLowess_*.tar.gz
	@echo "R package install complete!"

install-dev:
	@echo "Installing R package from source (development mode)..."
	@Rscript -e "devtools::install()"
	@echo "Development install complete!"

load:
	@echo "Loading R package for testing..."
	@Rscript -e "devtools::load_all('.')"
	@echo "Package loaded!"

document:
	@echo "Generating R documentation..."
	@Rscript -e "roxygen2::roxygenise()"
	@echo "Documentation generated!"

# --- Test ---
test: test-r

test-rust:
	@echo "Running Rust tests..."
	@cargo test
	@echo "Rust tests complete!"

test-r:
	@echo "Running R examples..."
	@Rscript demo/batch_smoothing.R
	@Rscript demo/online_smoothing.R
	@Rscript demo/streaming_smoothing.R
	@echo "R examples complete!"

test-all: test-rust test-r
	@echo "All tests complete!"

# --- R CMD check ---
check-r:
	@echo "Running R CMD check..."
	@R CMD build .
	@R CMD check fastLowess_*.tar.gz
	@echo "R CMD check complete!"

check-r-no-manual:
	@echo "Running R CMD check (without manual)..."
	@R CMD build .
	@R CMD check --no-manual fastLowess_*.tar.gz
	@R CMD check --no-manual fastLowess_*.tar.gz
	@echo "R CMD check complete!"

# --- Release ---
cran:
	@echo "Preparing for CRAN submission..."
	@./scripts/prepare_cran.sh
	@$(MAKE) install
	@echo "DONE! Package tarball is ready for submission."

# --- Documentation ---
doc: doc-default doc-serial
	@echo "Documentation build complete!"

doc-default:
	@echo "Building Rust documentation (default / parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

doc-serial:
	@echo "Building Rust documentation (serial / no parallel)..."
	@RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --no-default-features

# --- Clean ---
clean: clean-rust clean-r
	@echo "Clean complete!"

clean-rust:
	@echo "Performing cargo clean..."
	@cargo clean

clean-r:
	@echo "Cleaning R build artifacts..."
	@rm -rf fastLowess.Rcheck
	@rm -rf fastLowess_*.tar.gz
	@rm -rf src/*.o
	@rm -rf src/*.so
	@rm -rf src/fastLowess.so
	@rm -rf fastLowess*.tar.gz
	@rm -rf target

# --- Help ---
help:
	@echo "Available targets:"
	@echo "  check          - Run all Rust checks (fmt, clippy, build, test, doc)"
	@echo "  check-all      - Run all checks (Rust + R linting)"
	@echo ""
	@echo "Formatting:"
	@echo "  fmt            - Check Rust formatting"
	@echo "  fmt-fix        - Fix Rust formatting"
	@echo ""
	@echo "Linting:"
	@echo "  clippy         - Run Rust clippy"
	@echo "  lint-r         - Run R lintr"
	@echo ""
	@echo "Building:"
	@echo "  build          - Build Rust crate"
	@echo "  install        - Build and install R package"
	@echo "  install-dev    - Install R package (development mode)"
	@echo "  load           - Load R package for testing"
	@echo "  document       - Generate R documentation with roxygen2"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run R example scripts"
	@echo "  test-rust      - Run Rust tests"
	@echo "  test-r         - Run R examples"
	@echo "  test-all       - Run all tests"
	@echo ""
	@echo "R Package:"
	@echo "  check-r        - Run R CMD check"
	@echo "  check-r-no-manual - Run R CMD check (skip PDF manual)"
	@echo "  cran           - Prepare for CRAN submission (offline build)"
	@echo ""
	@echo "Other:"
	@echo "  doc            - Build Rust documentation"
	@echo "  clean          - Clean all build artifacts"
	@echo "  help           - Show this help message"
