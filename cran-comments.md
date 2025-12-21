# fastlowess

## Test environments

- Local Linux (Ubuntu/Debian)
- GitHub Actions: Ubuntu-latest, macOS-latest, Windows-latest

## R CMD check results

0 errors | 0 warnings | 3 notes
(Note: 1 warning about missing checkbashisms script in some environments)

## Submissions

This is a new submission.

## Comments

- Rust dependencies are vendored in `src/vendor/` for self-contained builds.
- Remaining NOTEs about hidden files in `src/vendor/` are essential cargo metadata (`.cargo-checksum.json`) required for successful compilation from source.
- The NOTE about the `-march=nocona` flag is inherited from the system's R configuration and is not explicitly requested by the package.
- The `src/.cargo` directory was renamed to `src/cargo` to avoid hidden file warnings, with build configuration passed via `--config`.
- The `configure` script handles platform-specific linker flags for Linux and macOS.
