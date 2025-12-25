# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- Added the option of installing from R-universe without needing Rust.

### Changed

- Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0.
- Updated cross-validation API to use tuple constructors (`KFold`, `LOOCV`).

## [0.2.0]

- For changes to the core logic and the API, see the [lowess](https://github.com/av746/lowess) and [fastLowess](https://github.com/av746/fastLowess) crates.

### Added

- Added support for new features in `fastLowess` v0.2.0.

### Changed

- Improved documentation.

## [0.1.0]

### Added

- Added the R binding for `fastLowess`.
