# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Bumped envisim-dependencies.


## [0.3.1] - 2026-06-02
- Bumped envisim-dependencies.


## [0.3.0] - 2026-05-25
- Minimum R version 4.6
- MSRV: 1.85.1
- Bumped envisim-dependencies.

This release breaks historical seeds.

- Fixed bug in `brewer` unequal probability sampling.
- Fixed bug in `cube`.
- Added `sequential_cube`, which determines units in the provided order. `cube`, on the other hand, determines units in a random order, and uses random windows in the flight phase. (#28)


## [0.2.0] - 2026-03-31
- Added distributionally balanced designs `dbd_tc`, `dbd_circular`
- Fixed bug in `cube`, `local_cube` (thanks Arvi!)
- Removed `seed` option -- use `R`s `unif_rand`
- Build script checks Macos SDK version and exports MACOSX_DEPLOYMENT_TARGET if not set by CRAN, see e.g. [clarabel-r](https://github.com/oxfordcontrol/clarabel-r/commit/a3d5ec6a5432427a77df88b4d4474c1aef244e1e)


## [0.1.0] - 2025-08-15
Initial release.
