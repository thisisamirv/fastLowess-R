# fastlowess

## Test environments

- Local Linux (Ubuntu/Debian)
- GitHub Actions: Ubuntu-latest, macOS-latest, macOS-14 (ARM64), Windows-latest

## R CMD check results

0 errors | 0 warnings | 1 note

### NOTE: Hidden files in src/vendor/

The `.cargo-checksum.json` files in `src/vendor/` are required for vendored Rust dependencies. Cargo uses these files to verify dependency integrity during compilation. They cannot be removed without breaking the build.

## Submissions

This is a new submission.

## Comments

- This package provides Rust-powered LOWESS smoothing with parallel execution.
- Rust dependencies are vendored in `src/vendor/` for self-contained builds.
- The `configure` script handles platform-specific linker flags for Linux and macOS.
- The `src/.cargo` directory was renamed to `src/cargo` to avoid hidden file warnings.
