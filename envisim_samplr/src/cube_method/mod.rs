// Copyright (C) 2026 Wilmer Prentius.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Cube methods
//!
//! Implements [`CubeSampling`] for [`SamplingOptions`].
//!
//! # References
//! Chauvet, G. (2009).
//! Stratified balanced sampling.
//! Survey Methodology, 35(1), 115-119.
//!
//! Deville, J. C., & Tillé, Y. (2004).
//! Efficient balanced sampling: the cube method.
//! Biometrika, 91(4), 893-912.
//! <https://doi.org/10.1093/biomet/91.4.893>
//!
//! Grafström, A., & Tillé, Y. (2013).
//! Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
//! Environmetrics, 24(2), 120-131.
//! <https://doi.org/10.1002/env.2194>
//!
//! Leuenberger, M., Eustache, E., Jauslin, R., & Tillé, Y. (2022).
//! Balancing a sample almost perfectly.
//! Statistics & Probability Letters, 180, 109229.
//! <https://doi.org/10.1016/j.spl.2021.109229>

mod cube;
mod stratified;
mod utils;

pub use cube::{
    CubeRunner,
    CubeSampling,
    CubeStrategy,
    LocalCubeSampling,
    LocalCubeStrategy,
    RandomCubeStrategy,
    SequentialCubeStrategy,
};
pub use envisim_utils::sampling_options::SamplingOptions;
pub use stratified::{
    CubeStratifiedRunner,
    cube_stratified,
    local_cube_stratified,
};

pub use crate::error::SamplingError;
