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

//! Sampling algorithms based on the pivotal method
//!
//! Implements [`PivotalSampling`] and [`LocalPivotalSampling`] for [`SamplingOptions`].
//!
//! # References
//! Deville, J. C., & Tille, Y. (1998).
//! Unequal probability sampling without replacement through a splitting method.
//! Biometrika, 85(1), 89-101.
//! <https://doi.org/10.1093/biomet/85.1.89>
//!
//! Grafström, A., Lundström, N. L., & Schelin, L. (2012).
//! Spatially balanced sampling through the pivotal method.
//! Biometrics, 68(2), 514-520.
//! <https://doi.org/10.1111/j.1541-0420.2011.01699.x>

mod base;
mod runner;
mod spatial;

pub use base::{
    PivotalSampling,
    RandomStrategy,
    SequentialStrategy,
};
pub use envisim_utils::sampling_options::SamplingOptions;
pub use runner::{
    PivotalRunner,
    PivotalStrategy,
};
pub use spatial::{
    LocalPivotalSampling,
    LocalStrategy1,
    LocalStrategy1S,
    LocalStrategy2,
    hierarchical_lpm_2,
};

pub use crate::error::SamplingError;
