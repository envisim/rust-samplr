# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
