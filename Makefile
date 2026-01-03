# Configuration
PKG_NAME := rfastlowess
PKG_VERSION := $(shell grep "^Version:" DESCRIPTION | sed 's/Version: //')
PKG_TARBALL := $(PKG_NAME)_$(PKG_VERSION).tar.gz

# Standard dev check
check: vendor fmt build install doc test submission

# ============================================================================== #
# Vendor
# ============================================================================== #
vendor: vendor-update extract-vendor

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

extract-vendor:
	@if [ -f src/vendor.tar.xz ] && [ ! -d src/vendor ]; then \
		echo "Extracting vendored dependencies..."; \
		(cd src && tar --extract --xz -f vendor.tar.xz); \
	fi

# ==============================================================================
# Formatting
# ==============================================================================
fmt: fmt-fix fmt-check clippy lint-r

fmt-fix:
	@(cd src && cargo fmt --all)

fmt-check:
	@echo "Rust formatting..."
	@(cd src && cargo fmt --all -- --check || (echo "Run 'make fmt-fix' to fix"; exit 1))

clippy:
	@echo "Rust clippy..."
	@(cd src && cargo clippy --all-targets -- -D warnings)

lint-r:
	@echo "R linting..."
	@Rscript -e "lintr::lint_package()" || true

# ==============================================================================
# Build
# ==============================================================================
build: build-rust build-r

build-rust:
	@echo "Building Rust crate..."
	@(cd src && cargo build --release)

build-r: doc-r
	@echo "Building R package tarball..."
	R CMD build .

# ==============================================================================
# Install
# ==============================================================================
install: install-r install-dev

install-r:
	@echo "Installing R package..."
	R CMD INSTALL $(PKG_TARBALL)

install-dev:
	@echo "Installing R package (devtools mode)..."
	Rscript -e "devtools::install()"

# ==============================================================================
# Documentation
# ==============================================================================
doc: doc-rust doc-r vignettes

doc-rust:
	@echo "Generating Rust docs..."
	@(cd src && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps)

doc-r:
	@echo "Generating R docs (roxygen2)..."
	Rscript -e "devtools::document()"

vignettes:
	@echo "Building vignettes..."
	Rscript -e "devtools::build_vignettes()"

# ==============================================================================
# Testing
# ==============================================================================
test: test-rust test-r test-cran-emulation

test-rust:
	@echo "Running Rust tests..."
	@(cd src && cargo test)

test-r:
	@echo "Running R tests (devtools)..."
	Rscript -e "Sys.setenv(NOT_CRAN='true'); devtools::test()"

test-cran-emulation:
	@echo "Running R tests as CRAN would (skipping protected)."
	Rscript -e "devtools::test()"

# ==============================================================================
# Submission Checks
# ==============================================================================
submission: check-cran bioccheck size wasm-build

check-cran: build-r
	@echo "Running R CMD check --as-cran..."
	R_MAKEVARS_USER=$(PWD)/scripts/Makevars.check R CMD check --as-cran $(PKG_TARBALL)

bioccheck: build-r
	@echo "Running BiocCheck..."
	Rscript -e "if (!requireNamespace('BiocCheck', quietly = TRUE)) BiocManager::install('BiocCheck', ask=FALSE, update=FALSE); BiocCheck::BiocCheck('$(PKG_TARBALL)')"

size: build-r
	@echo "Package size (Limit: 5MB):"
	@ls -lh $(PKG_TARBALL)

wasm-build:
	@echo "Building WASM binary using R-universe Docker container..."
	@if ! command -v docker &> /dev/null; then \
		echo "Error: Docker is required. Install with: sudo pacman -S docker"; \
		exit 1; \
	fi
	@if ! docker image inspect ghcr.io/r-universe-org/build-wasm:latest &> /dev/null; then \
		echo "Pulling R-universe Docker image (this may take a while)..."; \
		docker pull ghcr.io/r-universe-org/build-wasm:latest; \
	fi
	docker run --rm \
		-v "$(PWD)":/pkg \
		-w /pkg \
		ghcr.io/r-universe-org/build-wasm:latest \
		bash -c "R -e \"rwasm::build('./$(PKG_TARBALL)')\" && echo 'WASM build successful!'"

# ==============================================================================
# Other
# ==============================================================================
clean: clean-wasm clean-rust clean-r clean-other

clean-wasm:
	@if [ -d src/target ]; then \
		if rm -rf src/target 2>/dev/null; then \
			true; \
		elif command -v docker >/dev/null; then \
			docker run --rm -v "$(PWD)":/pkg ghcr.io/r-universe-org/build-wasm:latest rm -rf /pkg/src/target; \
		else \
			echo "Warning: Failed to clean src/target (permissions) and Docker not found."; \
		fi \
	fi

clean-rust:
	@(cd src && cargo clean 2>/dev/null || true)
	@rm -rf src/vendor
	@rm -rf src/Cargo.lock
	@rm -rf target

clean-r:
	@rm -rf $(PKG_NAME).Rcheck
	@rm -rf $(PKG_NAME).BiocCheck
	@rm -f $(PKG_NAME)_*.tar.gz
	@rm -rf src/*.o src/*.so src/*.dll
	@rm -rf doc Meta vignettes/*.html
	@find . -name "*.Rout" -delete

clean-other:
	@rm -rf src/Makevars

coverage:
	@echo "Calculating R coverage..."
	Rscript -e "if (!requireNamespace('covr', quietly = TRUE)) install.packages('covr'); Sys.setenv(NOT_CRAN='true'); covr::package_coverage()"
