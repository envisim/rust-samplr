# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-06-02
- Bumped dependency `envisim_utils`.


## [0.6.0] - 2026-05-25
- MSRV: 1.85.1
- Added dependency [`num-traits`](https://crates.io/crates/num-integer).
- Added dependency [`num-traits`](https://crates.io/crates/num-traits).
- Bumped dependency `envisim_utils`.

- Fixed bug in `brewer` unequal probability sampling.
- Fixed bug in `cube`.
- Refactored sampling methods to use new `SamplingOptions` and `SampleController`. Infallible methods are now infallible.
- As coordination is removed from `SamplingOptions`, `cps_coord` and `scps_coord` dedicated methods now accept random values for coordination.
- Added `sequential_cube`, which determines units in the provided order. `cube`, on the other hand, determines units in a random order, and uses random windows in the flight phase. (#28)
- Moved dbd evaluating methods to its own trait, and renamed the methods.
- Sampling methods use new RNG traits from `envisim_utils`.


## [0.5.0] - 2026-03-27
- Bump `envisim_utils`.
- Refactored sampling functions, now uses SamplingOptions from `envisim_utils`.
- SamplingError, removed Input, Indices, Node; added Options, IncorrectStratification, IncorrectDrawProbabilities, IncorrectProbabilitiesIntegerSum.
- Removed SampleContainer, added SampleController.
- Fix a bug in cube null space.
- Added Distributionally balanced designs.


## [0.4.1] - 2026-02-26
- Fix a bug in cube null space


## [0.4.0] - 2025-08-22
### Changed
- Update `rustc-hash` dependency to 2.1.1.

### Removed
- `rand` dependency.


## [0.3.0] - 2025-08-15
### Changed
Breaking changes:

- `SamplingOptions` interface use getter and setter.
- `SamplingOptions` move self on setter, instead of by reference.
- `SamplingOptions` can set spreading by `AuxiliariesOptions`.
- `poisson` module removed in favour of unequal module (conditional poisson, poisson) and correlated_poisson module.
- Refactored sampling methods to use variant structs.
- `conditional_poisson` early return for sample size 0.


## [0.2.0] - 2024-09-24
### Added
- moved `SamplingError`, previously available from `envisim_utils`.
- re-exports `SamplingError` from sub modules

### Changed
- `SamplingError` does not depend on `thiserror`
- maximum iterations is now of type `NonZeroUsize`.
- maximum iterations now set through `SampleOptions`.
- `SamplingError::MaxIterations` changed to `SamplingError::MaxIterations(NonZeroUsize)`.
- `InputError`and `Probabilities` no longer exported from `poisson` module

### Removed
- removed unused `SamplingError::General`.


## [0.2.0] - 2024-09-24
### Added
- moved `SamplingError`, previously available from `envisim_utils`.
- re-exports `SamplingError` from sub modules

### Changed
- `SamplingError` does not depend on `thiserror`
- maximum iterations is now of type `NonZeroUsize`.
- maximum iterations now set through `SampleOptions`.
- `SamplingError::MaxIterations` changed to `SamplingError::MaxIterations(NonZeroUsize)`.
- `InputError`and `Probabilities` no longer exported from `poisson` module

### Removed
- removed unused `SamplingError::General`.


## [0.1.0] - 2024-09-19
Initial release.
