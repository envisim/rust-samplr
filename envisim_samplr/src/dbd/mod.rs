// Copyright (C) 2026 Wilmer Prentius
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

//! Distributionally balanced designs.
//!
//! Implements [`DistributionalDesigns`] for [`SamplingOptions`].
//!
//! # References
//! Grafström, A., & Prentius, W. (2026).
//! Distributionally balanced sampling designs.
//! arXiv preprint arXiv:2603.11916.
//! <https://doi.org/10.48550/arXiv.2603.11916>
//!
//! Grafström, A., & Prentius, W. (2026).
//! Distributionally balanced sampling designs via minimum tactical configurations.
//! arXiv preprint arXiv:2603.24439.
//! <https://doi.org/10.48550/arXiv.2603.24439>

mod annealing;
mod dbd_circular;
mod dbd_circular_config;
mod dbd_options;
mod dbd_tc;
mod dbd_tc_config;
mod options_impl;
mod tc_parameters;

pub use dbd_circular::CircularConfiguration;
pub use dbd_options::DistributionalDesignOptions;
pub use dbd_tc::TacticalConfiguration;
pub use envisim_utils::sampling_options::SamplingOptions;
pub use options_impl::{
    DistributionalDesignEvaluators,
    DistributionalDesigns,
};
pub use tc_parameters::{
    DbdConfiguration,
    TacticalConfigurationParameters,
};

pub use crate::SamplingError;
