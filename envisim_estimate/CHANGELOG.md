# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `balance_deviation`, to calculate deviation between total and total estimate.

### Changed
- `local_mean_variance`, `spatial_balance::local`, `spatial_balance::voronoi` use `SampleOptions`
instead of `TreeBuilder`.
- `nearest_neighbour` use `Matrix` instead of `TreeBuilder`.
- `local_mean_variance`, `nearest_neighbour`, `spatial_balance::local`, `spatial_balance::voronoi`
returns `f64::NAN` on empty sample.
- bug in `local_mean_variance`.

## [0.2.0] - 2024-09-24
### Added
- added dependency `envisim_samplr`.

### Changed
- `n_neighbours` parameter of `local_mean_variance` changed type from `usize` to `NonZeroUsize`.

### Removed
- removed dependency `thiserror`.
- removed re-exports from `spatial_balance` module.

## [0.2.0] - 2024-09-24
### Added
- added dependency `envisim_samplr`.

### Changed
- `n_neighbours` parameter of `local_mean_variance` changed type from `usize` to `NonZeroUsize`.

### Removed
- removed dependency `thiserror`.
- removed re-exports from `spatial_balance` module.

## [0.1.0] - 2024-09-19
Initial release.
