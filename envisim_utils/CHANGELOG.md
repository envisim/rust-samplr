# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0 - 2026-03-27]
- Matrix use std::borrow instead of reinventing the wheel.
- MatrixIndex as struct instead of tuple.
- Most Matrix methods are allowed to fail by returning Option.
- Matrix::dim renamed to Matrix::dims
- Added Matrix::distance_between_rows
- Remove reexports from modules
- Added SamplingOptions (previously SampleOptions in `envisim_samplr`)
- Removed struct Probabilities -- added trait Probabilities and sturcts ProbabilitiesEqual, ProbabilitiesUnequal
- Added Indices::seq_after
- Removed InputError, added NodeSearcherError, PipsError.
- removed utils::sum, added utils::f64_to_usize

## [0.3.0] - 2025-08-22
### Changed
- Changed `rand` into an optional dependency, activated by feature `rand` (default).
- Update `rand` dependency to 0.9.2.
- Update `rustc-hash` dependency to 2.1.1.

## [0.2.1] - 2025-08-15
- Fixed elided lifetime warnings.
- Added `InputError::IsNone` error.

## [0.2.0] - 2024-09-24
### Added
- `Searcher::new_1`, shorthand for `Searcher::new(.., 1)`.

### Changed
- `NodeError::GhostUnit` changed to `NodeError::GhostIndex(usize)`.
- `Searcher::new` returns `Self` instead of `Result`.
- `Searcher::set_n_neighbours` returns void instead of `Result`.
- `n_neighbours` of `searcher` changed type from `usize` to `NonZeroUsize`.

### Removed
- removed `SearcherError`.
- removed dependency `thiserror`.
- removed unused `NodeError::General`.
- removed unused `InputError::General`.
- removed unused `InputError::Node`.
- moved `SamplingError`, now available in `envisim_samplr`.


## [0.1.0] - 2024-09-19
Initial release.
