# Contributing to fastLowess (R)

Thank you for your interest in contributing to `fastLowess`! We welcome bug reports, feature suggestions, documentation improvements, and code contributions.

## Quick Links

- 🐛 [Report a bug](https://github.com/thisisamirv/fastLowess-R/issues/new?labels=bug)
- 💡 [Request a feature](https://github.com/thisisamirv/fastLowess-R/issues/new?labels=enhancement)
- 📖 [Documentation](https://github.com/thisisamirv/fastLowess-R)
- 💬 [Discussions](https://github.com/thisisamirv/fastLowess-R/discussions)

## Code of Conduct

Be respectful, inclusive, and constructive. We're here to build great software together.

## Reporting Bugs

**Before submitting**, search existing issues to avoid duplicates.

Please include:

- **Clear description** of the problem.
- **Minimal Reproducible Example (reprex)**.
- **Environment details**: `sessionInfo()` output in R.
- **Expected vs Actual behavior**.

**Example:**

```r
library(fastLowess)

# This produces unexpected output
x <- c(1, 2, 3)
y <- c(1, 2, 3)
smooth(x, y, fraction = 0.5)
# Expected: ...
# Actual: ...
```

## Suggesting Features

Feature requests are welcome! Please:

- **Check existing issues** first.
- **Explain the use case** - why is this needed?
- **Provide examples** of how it would work in R.
- **Consider alternatives** - have you tried existing parameters?

Areas of particular interest:

- Performance optimizations.
- Better error messages.
- New kernels or robustness methods.

## Pull Requests

### Process

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with clear, focused commits.

3. **Add tests/examples**:
   - Add a new demo in `demo/` if it's a major feature.
   - Add examples in `R/` files (roxygen comments) for functions.

4. **Ensure quality**:

   ```bash
   make check-all   # Runs Rust checks + R linting
   make check-r     # Runs R CMD check
   ```

5. **Submit PR** with a clear description of changes.

### PR Checklist

- [ ] `make check-r` passes with **0 errors, 0 warnings**.
- [ ] `make lint-r` passes (clean code style).
- [ ] Documentation updated (`make document` run).
- [ ] `cargo fmt` applied (if Rust code changed).
- [ ] `cargo clippy` passes (if Rust code changed).
- [ ] New features have examples or demos.
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

## Development Setup

This project uses [rextendr](https://github.com/extendr/rextendr) to bind Rust code to R.

### Prerequisites

- **Rust**: Latest stable (`rustup update`).
- **R**: 4.0+
- **R Packages**: `devtools`, `roxygen2`, `lintr`.

### Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fastLowess-R.git
cd fastLowess-R

# Install R dependencies
Rscript -e 'install.packages(c("devtools", "roxygen2", "lintr"))'

# Install project in development mode
make install
```

### Development Commands

We use a `Makefile` to simplify common tasks:

```bash
# --- Build & Install ---
make install      # Build and install package
make document     # Generate .Rd documentation (roxygen2)

# --- Check & Test ---
make test-r       # Run the demos/examples
make check-r      # Run full R CMD check
make lint-r       # Check R code style
make fmt          # Format Rust code
make clippy       # Check Rust code quality

# --- Clean ---
make clean        # Remove build artifacts
```

## Project Structure

```text
fastLowess-R/
├── R/                  # R wrapper functions
├── src/                # Rust backend code
│   └── lib.rs          # extendr bindings
├── demo/               # Interactive R demos (Testing scripts)
├── inst/               # Installed files (CITATION, AUTHORS)
├── man/                # Generated documentation (.Rd) - DO NOT EDIT MANUALLY
├── scripts/            # Helper scripts (prepare_cran.sh)
├── Cargo.toml          # Rust dependencies
└── DESCRIPTION         # R package metadata
```

## Testing Guidelines

We rely on **examples and demos** for verification.

### Running Tests

```bash
# Run all demos as a test suite
make test-r
```

### Writing Tests

1. **Demos**: Create a `.R` script in `demo/` that sets up data, runs the function, and asserts/prints results.
2. **Examples**: Add `@examples` blocks in your roxygen comments in `R/*.R`. These are checked by `R CMD check`.

## Code Style

### Rust Code

- Format with `cargo fmt`.
- Follow strict clippy lints (`make clippy`).

### R Code

- Follow standard R style (tidyverse-like).
- Use `make lint-r` to verify compliance.

### Documentation

- All external functions must have **roxygen2** documentation.
- Update `DESCRIPTION` if you add dependencies.

## Release Process (CRAN)

1. **Update Version**: In `DESCRIPTION` and `Cargo.toml`.
2. **Update Notes**: Edit `cran-comments.md`.
3. **Prepare Tarball** (The most significant step):

   Running `make cran` automates the release packaging. It:
   - Vendors all Rust dependencies into `vendor/`.
   - Creates a local `.cargo/config.toml`.
   - Generates `inst/AUTHORS`.
   - Builds the `.tar.gz`.

   ```bash
   make cran
   ```

4. **Submit**:
   Upload the resulting `fastLowess_*.tar.gz` to CRAN.

## License

By contributing, you agree that your contributions will be licensed under the **AGPL-3.0 License**.
