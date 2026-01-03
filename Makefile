# Makefile for fastlowess
# Unifies Rust development and R package verification

.PHONY: all help \
	check check-all check-cran bioccheck \
	build build-rust build-r \
	test test-rust test-r test-cran-emulation \
	doc doc-rust doc-r vignettes \
	clean clean-rust clean-r \
	install install-dev \
	fmt fmt-fix clippy lint-r \
	submit-ready vendor-update size coverage

# ==============================================================================
# Configuration
# ==============================================================================

PKG_NAME := fastlowess
PKG_VERSION := $(shell grep "^Version:" DESCRIPTION | sed 's/Version: //')
PKG_TARBALL := $(PKG_NAME)_$(PKG_VERSION).tar.gz

# Default target
all: check

# ==============================================================================
# Main Targets
# ==============================================================================

help:
	@echo "fastlowess Makefile"
	@echo "-------------------"
	@echo "Development:"
	@echo "  make check         - Run standard checks (Rust fmt/clippy, R lint, basic build)"
	@echo "  make check-all     - Run ALL checks including R CMD check and BiocCheck"
	@echo "  make build         - Build Rust crate and R package"
	@echo "  make test          - Run all tests (Rust + R)"
	@echo "  make doc           - Generate all documentation"
	@echo "  make install       - Install package locally"
	@echo "  make clean         - Remove all artifacts"
	@echo ""
	@echo "R Package Verification:"
	@echo "  make check-cran    - Run R CMD check --as-cran"
	@echo "  make bioccheck     - Run Bioconductor checks"
	@echo "  make submit-ready  - Full verification sequence for submission"
	@echo "  make size          - Check package tarball size"
	@echo "  make coverage      - Calculate test coverage"
	@echo ""
	@echo "Rust Specific:"
	@echo "  make fmt           - Check Rust formatting"
	@echo "  make fmt-fix       - Fix Rust formatting"
	@echo "  make clippy        - Run Rust linter"
	@echo "  make vendor-update - Update vendored dependencies"

# ==============================================================================
# Building & Installation
# ==============================================================================

build: build-rust build-r

# Extract vendor from tar.xz if needed (for local development)
vendor-extract:
	@if [ -f src/vendor.tar.xz ] && [ ! -d src/vendor ]; then \
		echo "Extracting vendored dependencies..."; \
		(cd src && tar --extract --xz -f vendor.tar.xz); \
	fi

# Build Rust crate (release mode by default for package)
build-rust: vendor-extract
	@echo "Building Rust crate..."
	@(cd src && cargo build --config cargo/config.toml --release)

# Build R package tarball
build-r: doc-r
	@echo "Building R package tarball..."
	R CMD build .

install: build-r
	@echo "Installing R package..."
	R CMD INSTALL $(PKG_TARBALL)

install-dev:
	@echo "Installing R package (devtools mode)..."
	Rscript -e "devtools::install()"

# ==============================================================================
# Checks & Verification
# ==============================================================================

# Standard dev check
check: fmt clippy build test lint-r

# Full verification suite
check-all: check check-cran bioccheck

# R CMD check
check-cran: build-r
	@echo "Running R CMD check --as-cran..."
	R_MAKEVARS_USER=$(PWD)/scripts/Makevars.check R CMD check --as-cran $(PKG_TARBALL)

# BiocCheck
bioccheck: build-r
	@echo "Running BiocCheck..."
	Rscript -e "if (!requireNamespace('BiocCheck', quietly = TRUE)) BiocManager::install('BiocCheck', ask=FALSE, update=FALSE); BiocCheck::BiocCheck('$(PKG_TARBALL)')"

# Full submission readiness workflow
submit-ready: clean doc-r build-r check-cran bioccheck
	@echo ""
	@echo "✅ Submission Verification Complete set for: $(PKG_TARBALL)"
	@echo "Next: Review logs in $(PKG_NAME).Rcheck/ and submit."

# Quick check (skipping manual and vignettes for speed)
quick-check:
	R CMD build --no-build-vignettes .
	R CMD check --no-build-vignettes --as-cran $(PKG_TARBALL)

# Package size check
size: build-r
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(PKG_TARBALL)

# ==============================================================================
# Testing
# ==============================================================================

test: test-rust test-r

test-rust:
	@echo "Running Rust tests..."
	@(cd src && cargo test --config cargo/config.toml)

test-r:
	@echo "Running R tests (devtools)..."
	Rscript -e "Sys.setenv(NOT_CRAN='true'); devtools::test()"

test-cran-emulation:
	@echo "Running R tests as CRAN would (skipping protected)."
	Rscript -e "devtools::test()"

coverage:
	@echo "Calculating R coverage..."
	Rscript -e "if (!requireNamespace('covr', quietly = TRUE)) install.packages('covr'); Sys.setenv(NOT_CRAN='true'); covr::package_coverage()"

# ==============================================================================
# Code Quality (Rust & R)
# ==============================================================================

fmt: vendor-extract
	@echo "Rust formatting..."
	@(cd src && cargo fmt --all -- --check || (echo "Run 'make fmt-fix' to fix"; exit 1))

fmt-fix: vendor-extract
	@(cd src && cargo fmt --all)

clippy: vendor-extract
	@echo "Rust clippy..."
	@(cd src && cargo clippy --config cargo/config.toml --all-targets -- -D warnings)

lint-r:
	@echo "R linting..."
	@Rscript -e "lintr::lint_package()" || true

# ==============================================================================
# Documentation
# ==============================================================================

doc: doc-rust doc-r vignettes

doc-rust:
	@echo "Generating Rust docs..."
	@(cd src && RUSTDOCFLAGS="-D warnings" cargo doc --config cargo/config.toml --no-deps)

doc-r:
	@echo "Generating R docs (roxygen2)..."
	Rscript -e "devtools::document()"

vignettes:
	@echo "Building vignettes..."
	Rscript -e "devtools::build_vignettes()"

# ============================================================================== #
# Vendor Update
# ============================================================================== #

vendor-update:
	@echo "Updating and re-vendoring crates.io dependencies..."
	@sed -i 's|fastLowess = { path = "vendor/fastLowess",|fastLowess = {|g' src/Cargo.toml
	@sed -i '/\[patch.crates-io\]/,/lowess = { path = "vendor\/lowess" }/d' src/Cargo.toml
	@rm -rf src/vendor src/vendor.tar.xz
	@(cd src && cargo vendor vendor)
	@./scripts/clean_checksums.py src/vendor
	@sed -i 's|fastLowess = {|fastLowess = { path = "vendor/fastLowess",|g' src/Cargo.toml
	@echo "" >> src/Cargo.toml
	@echo "[patch.crates-io]" >> src/Cargo.toml
	@echo "lowess = { path = \"vendor/lowess\" }" >> src/Cargo.toml
	@echo "Creating vendor.tar.xz archive..."
	@(cd src && tar --sort=name --mtime='1970-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner --xz --create --file=vendor.tar.xz vendor)
	@rm -rf src/vendor
	@echo "Vendor update complete. Archive: src/vendor.tar.xz"

# ==============================================================================
# Maintenance
# ==============================================================================

clean: clean-rust clean-r

clean-rust:
	@(cd src && cargo clean 2>/dev/null || true)
	@rm -rf src/vendor
	@rm -rf src/Cargo.lock
	@rm -rf target

clean-r:
	@rm -rf $(PKG_NAME).Rcheck
	@rm -f $(PKG_NAME)_*.tar.gz
	@rm -rf src/*.o src/*.so src/*.dll src/target
	@rm -rf doc Meta vignettes/*.html
	@find . -name "*.Rout" -delete
