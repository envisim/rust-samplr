# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `Matrix::swap`, for swapping two elements in a `Matrix`.

### Changed
- Renamed methods in `PointSet`. Use prefix `get_` rather than `try_`, renamed `exists` to `contains`, `size` to `len`, `dim` to `dimensions`, `id_iter` to `ids`. Added methods `coords` and `get_coords`, returning an iterator over an id.
- For searchers in `kd_tree::searcher`, renamed `from_slice`, `set_from_slice`, `reset_from_slice` to `_point`. These methods now accepts iterators instead of slices.
- Matrix storage: Removed `OwnedMatrixData`, `BorrowedMatrixData` and added `RawDataMut`. Implemented `RawData` and `RawDataMut` for array-like types in `std`.

### Removed
- Removed second (ergonomic) generic for data type in `Matrix`. Defaulted to `RawData::Elem`, now derived from the same.

## [0.6.0] - 2026-06-02
### Changed
- Added `fmt::Debug` as a supertrait of `Number`.
- Changed `PointSet` to be generic over `Id` and `Value`, previously only `Value` (renamed from `N`).
- Changed `FindSplit::split`, `MidpointSlide::new`, `Neighbour`, `WeightedNeighbour`, `WeightCollection` to be generic over `units` or `Id`.
- Changed `TreeSearcher`, `SearchPoint`, `NearestNeighbourSearcher`, `KNearestNeighbourSearcher`, `WeightedSearcher`, `Node`, `Leaf`, `Branch` to be generic over `PointSet`.
- Fixed bug in `Border::from_data`.


## [0.5.0] - 2026-05-25
- MSRV: 1.85.1
- Added dependency [`num-traits`](https://crates.io/crates/num-integer).
- Added dependency [`num-traits`](https://crates.io/crates/num-traits).
- Added dependency [`rand_core`](https://crates.io/crates/rand_core).
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
- Added trait `FloatRng` which extends `rand_core::Rng` with the method `next_f64`, which is needed in order to construct `Rng` from the R `unif_rand` API.
- Added trait `Rand<N>`, which extends `Rng` to draw numbers from `N`. Implemented for primitive integers and `f64`.
- Added trait `RandSlice<N>` which extends `Rand<N>` to draw numbers into a slice.
- Added functions `random_element`, `random_weighted` for drawing a random element from a slice and making a weighted selection respectively.
- Added `Epsilon` wrapper for float-based comparisons.
- Added trait `ProbabilityOptions`, with implementors `EqualProbabilityOptions` and `UnequalProbabilityOptions`.
- Added `Probability`, as a wrapper around numbers that can be represented as probabilities.
- Added struct `ProbabilitySet`, replacing `ProbabilityStore`.

### Changed
- Made `Matrix` generic over `Number`. `Matrix` is now a type alias for `MatrixBase`, and represents an owned matrix. The type alias for a reference matrix is `MatrixRef`.
- Removed `MatrixIterator` in favour of slice implementation.
- Added `inverse` method to `Matrix`.
- Removed `Node`'s dependency on data, and transformed it into an enum, replacing `NodeKind`.
- Changed `TreeSearcher` trait
- Refactored `SamplingOptions` to follow the typestate pattern (again). The types are concrete members of the options object, and unset members are represented by `()`. In addition to the probability specification, spreading and balancing are represented in this pattern. Coordination is now not set through `SamplingOptions`, but still provided as a helper through `CoordinationOptions`.
- `ProbabilityStore` associated type now requires `num_traits::NumAssign`.
- Refactored `SampleController`. Instead of being a trait, it is now a generic struct using the typestate pattern, implementing special methods for when a kd-tree is present. When a tree is present, removing a unit requires it to be removed from the tree aswell. This is ensured through the trait `UnitRemoving`.
- Methods on `Indices` returns `bool` instead of `Option` where applicable.
- `Indices` initialises its store in reversed order.
- `SamplingOptions` to operate on `ProbabilityOptions'.
- `Sample` is exported from its own module `envisim_utils::sample::Sample`.

### Removed
- Removed `MatrixIndex`.
- Removed trait `TreeBuilder`, in favour of `TreeConfig`.
- Removed enum `NodeKind`.
- Removed tree-searchers `Searcher`, `SearcherWeigthed`.
- Removed type alias `FindSplit`.
- Removed utility functions `usize_to_f64` and `f64_to_usize`. Use `num_traits::ToPrimitive` instead.
- Removed trait `RandomNumberGenerator`.
- Removed `DecideUnit` in favour of `Probability`.
- Removed the trait `ProbabilitySpec`, and its implementors `ProbabilitySpecUnequal` and `ProbabilitySpecEqual`.
- Removed the trait `ProbabilityStore`, and its implementors `FloatProbabilities` and `ExactProbabilities`.


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
