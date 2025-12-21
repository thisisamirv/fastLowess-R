# Run all local checks (formatting, linting, building, tests, docs)
check: fmt clippy build test doc check-CMD cran
	@echo "All checks completed successfully!"

# Formatting
fmt: fmt-rust fmt-fix-rust
	@echo "Formatting check complete!"

fmt-rust:
	@echo "Checking Rust code formatting..."
	@(cd src && cargo fmt --all -- --check)

fmt-fix-rust:
	@echo "Formatting Rust code..."
	@(cd src && cargo fmt --all)

# Linter
clippy: clippy-default clippy-serial lint-r

clippy-default:
	@echo "Running clippy (default / parallel)..."
	@(cd src && cargo clippy --config cargo/config.toml --all-targets -- -D warnings)

clippy-serial:
	@echo "Running clippy (serial / no parallel)..."
	@(cd src && cargo clippy --config cargo/config.toml --all-targets --no-default-features -- -D warnings)
	@echo "Clippy check complete!"

lint-r:
	@echo "Linting R code with lintr..."
	@Rscript -e "lintr::lint_package()" || true
	@echo "R lint complete!"

# Build
build: build-default build-serial r-install r-install-dev

build-default:
	@echo "Building crate (default / parallel)..."
	@(cd src && cargo build --config cargo/config.toml)

build-serial:
	@echo "Building crate (serial / no parallel)..."
	@(cd src && cargo build --config cargo/config.toml --no-default-features)
	@echo "Build complete!"

r-install:
	@echo "Building and installing R package..."
	@R CMD build .
	@R CMD INSTALL fastlowess_*.tar.gz
	@echo "R package install complete!"

r-install-dev:
	@echo "Installing R package from source (development mode)..."
	@Rscript -e "devtools::install()"
	@echo "Development install complete!"

# Test
test:
	@echo "Running R examples..."
	@Rscript demo/batch_smoothing.R
	@Rscript demo/online_smoothing.R
	@Rscript demo/streaming_smoothing.R
	@echo "R examples complete!"

# R CMD check
check-CMD: check-r-no-manual
	@echo "All R CMD checks completed successfully!"

check-r-no-manual:
	@echo "Running R CMD check (without manual)..."
	@Rscript -e "devtools::check(manual = FALSE)"
	@echo "R CMD check complete!"

# Release
# CRAN submission preparation
cran:
	@echo "Preparing for CRAN submission..."
	@./scripts/prepare_cran.sh
	@$(MAKE) install
	@echo "DONE! Package tarball is ready for submission."

# Update vendored dependencies from crates.io
vendor-update:
	@echo "Updating dependencies from crates.io..."
	# 1. Switch to registry versions in Cargo.toml
	@sed -i 's|fastLowess = { path = "fastLowess",|fastLowess = {|g' src/Cargo.toml
	@sed -i '/\[patch.crates-io\]/,/lowess = { path = "lowess" }/d' src/Cargo.toml
	# 2. Remove local directories
	@rm -rf src/fastLowess src/lowess
	# 3. Re-vendor
	@echo "Running cargo vendor..."
	@(cd src && cargo vendor vendor)
	# 4. Update checksums (remove .gitignore references)
	@./scripts/clean_checksums.py src/vendor
	# 5. Restore path dependencies to point to vendor/
	@sed -i 's|fastLowess = {|fastLowess = { path = "vendor/fastLowess",|g' src/Cargo.toml
	@echo "" >> src/Cargo.toml
	@echo "[patch.crates-io]" >> src/Cargo.toml
	@echo "lowess = { path = \"vendor/lowess\" }" >> src/Cargo.toml
	@echo "Dependency update and re-vendoring complete!"

# Documentation
doc: doc-default doc-serial r-doc
	@echo "Documentation build complete!"

doc-default:
	@echo "Building Rust documentation (default / parallel)..."
	@(cd src && RUSTDOCFLAGS="-D warnings" cargo doc --config cargo/config.toml --no-deps)

doc-serial:
	@echo "Building Rust documentation (serial / no parallel)..."
	@(cd src && RUSTDOCFLAGS="-D warnings" cargo doc --config cargo/config.toml --no-deps --no-default-features)

r-doc:
	@echo "Generating R documentation..."
	@Rscript -e "roxygen2::roxygenise()"
	@echo "Documentation generated!"

# Clean
clean: clean-rust clean-r
	@echo "Clean complete!"

clean-rust:
	@echo "Performing cargo clean..."
	@(cd src && cargo clean)

clean-r:
	@echo "Cleaning R build artifacts..."
	@rm -rf fastlowess.Rcheck
	@rm -rf fastlowess_*.tar.gz
	@rm -rf src/*.o
	@rm -rf src/*.so
	@rm -rf src/fastlowess.so
	@rm -rf fastlowess*.tar.gz
	@rm -rf target
