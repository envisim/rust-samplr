# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
