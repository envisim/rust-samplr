# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
- fix: removed unused `SamplingError::General`.

## [0.1.0] - 2024-09-19
Initial release.
