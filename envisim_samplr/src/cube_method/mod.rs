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

mod cube;
mod stratified;
mod utils;

pub use cube::{
    BasicCubeStrategy,
    CubeRunner,
    CubeSampling,
    CubeStrategy,
    LocalCubeSampling,
    LocalCubeStrategy,
};
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
    SamplingOptionsResult,
    SpreadingOptions,
};
pub use stratified::{
    CubeStratifiedRunner,
    cube_stratified,
    local_cube_stratified,
};
