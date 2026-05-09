# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- MSRV: 1.85.1
- Added dependency [`num-traits`](https://crates.io/crates/num-traits).
- Bumped optional dependency [`rand`](https://crates.io/crates/rand) to 0.10.1.

### Added
- Added feature `"test-utils"`, which replaces `envisim_test_utils` private module, containing helpers for tests.
- Added trait `Number`, which requires `num_traits::NumAssign`. Implemented number for primitive ints and floats.
- Added trait `PointSet<Number>`, for accessing multidimensional unit data. Implemented for `Matrix`.
- Added trait `Dimensions`, implemented for `MatrixBase`.
- Added `MatrixDims` and `MatrixCoord`, replacing the dual-purpose `MatrixIndex`.
- Added `kd_tree::searcher::Neighbours` and `WeightedNeighbours` as a type.
- Added `Tree` as the `kd_tree` implementation. `Tree` is generic of `PointSet`, and stores a reference to the data used to construct the tree. Any method traversing the tree now have access to this reference, guaranteeing a relationship between the tree and its data.
- Added trait `TreeConfig` used to construct a tree. Implemented for `SpreadingOptions` (replaces `TreeBuilder`).
- Added tree-searchers `NearestNeighbourSearcher`, `KNearestNeighbourSearcher`, `WeightedSearcher`.
- Added neighbour-representations `Neighbour` and `WeightedNeighbour` used in tree-searchers.
- Added search point representation `SearchPoint`.
- Added trait `FindSplit`, implemented for `MidpointSlide`.
- Added `try_sys_rng` for `SmallRng`, which returns a random number generator using `rand::rngs::SysRng`.

### Changed
- Made `Matrix` generic over `Number`. `Matrix` is now a type alias for `MatrixBase`, and represents an owned matrix. The type alias for a reference matrix is `MatrixRef`.
- Refactored `MatrixIterator`.
- Removed `Node`'s dependency on data, and transformed it into an enum, replacing `NodeKind`.
- Changed `TreeSearcher` trait
- Refactored `SamplingOptions` to follow the typestate pattern (again). The types are concrete members of the options object, and unset members are represented by `()`. In addition to the probability specification, spreading and balancing are represented in this pattern. Coordination is now not set through `SamplingOptions`, but still provided as a helper through `CoordinationOptions`.
- `ProbabilityStore` associated type now requires `num_traits::NumAssign`.
- Refactored `SampleController`. Instead of being a trait, it is now a generic struct using the typestate pattern, implementing special methods for when a kd-tree is present. When a tree is present, removing a unit requires it to be removed from the tree aswell. This is ensured through the trait `UnitRemoving`.

### Removed
- Removed `MatrixIndex`.
- Removed trait `TreeBuilder`, in favour of `TreeConfig`.
- Removed enum `NodeKind`.
- Removed tree-searchers `Searcher`, `SearcherWeigthed`.
- Removed type alias `FindSplit`.
- Removed utility functions `usize_to_f64` and `f64_to_usize`. Use `num_traits::ToPrimitive` instead.


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
