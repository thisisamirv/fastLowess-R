## Test environments

- Local Linux (Ubuntu/Debian)
- GitHub Actions: Ubuntu-latest, macOS-latest, Windows-latest

## R CMD check results

0 errors | 0 warnings | 0 notes

## Submissions

This is a new submission.

## Comments

- Rust dependencies are vendored in the `vendor/` directory to satisfy CRAN's policy for offline builds.
- The `inst/AUTHORS` file lists the licenses and copyrights for all vendored Rust crates.
- The `configure` script handles platform-specific linker flags for Linux and macOS.
