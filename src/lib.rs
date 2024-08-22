// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

//! Design-based sampling methods, with a focus on spatially balanced and balanced sampling designs.

//! **Balanced sampling** utilizes auxilliary information in order to obtain a sample where the
//! Horvitz-Thompson (HT) estimator of the total of the auxilliary information equals the population
//! total of the auxilliaries.
//! This may be very efficient (yield relatively low variance) if there is a linear relationship
//! between the auxilliaries and the variable of interest.
//!
//! **Spatially balanced sampling** uses auxilliary information in order to obtain a sample that is
//! well-spread in auxilliary space, as well as being balanced.
//! The samples can then be seen as a miniature version of the population.
//! This generally yields low variances for the variable of interest, if there is a general
//! relationship between the auxilliaries and the variables of interest.

pub mod cube_method;
pub mod pivotal_method;
pub mod poisson;
pub mod srs;
pub mod systematic;
pub mod unequal;
mod utils;

mod sample_options;
pub use sample_options::{SampleOptions, Sampler};
