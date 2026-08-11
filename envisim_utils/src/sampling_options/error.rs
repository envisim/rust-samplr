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

//! Sampling options errors

use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SamplingOptionsError {
    #[error("eps must be in [0.0, 1.0)")]
    InvalidEpsilon,
    #[error("population size must be positive")]
    InvalidPopulationSize,
    #[error("sample contains invalid units")]
    InvalidSample,
    #[error("sample size must not be larger than the population size")]
    InvalidSampleSize,
    #[error("probabilities must be valid (nominally in [0.0, 1.0])")]
    InvalidProbability,
    #[error("the number of random values must not be smaller than the population size")]
    InvalidRandomValues,
    #[error("iterations must be positive")]
    InvalidIterations,
    #[error("the number of units in the spreading data must match the population size")]
    InvalidSpreading,
    #[error("bucket size must be at least 1")]
    InvalidBucketSize,
    #[error("the number of units the balancing data must match the population size")]
    InvalidBalancing,
    #[error("spreading data must be provided")]
    MissingSpreading,
    #[error("balancing data must be provided")]
    MissingBalancing,
}
pub type SamplingOptionsResult<T> = Result<T, SamplingOptionsError>;
