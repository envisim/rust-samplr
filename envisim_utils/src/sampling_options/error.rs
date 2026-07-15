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

/// Errors relating to the configuration of sampling algorithms
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SamplingOptionsError {
    /// Epsilon must be in [0.0, 1.0)
    #[error("eps must be in [0.0, 1.0)")]
    InvalidEpsilon,
    /// Population size must be positive
    #[error("population size must be positive")]
    InvalidPopulationSize,
    /// Sample contains invalid units
    #[error("sample contains invalid units")]
    InvalidSample,
    /// Sample size must not be larger than the population size
    #[error("sample size must not be larger than the population size")]
    InvalidSampleSize,
    /// Probabilities must be valid
    #[error("probabilities must be valid (nominally in [0.0, 1.0])")]
    InvalidProbability,
    /// The number of random values must not be smaller than the population size
    #[error("the number of random values must not be smaller than the population size")]
    InvalidRandomValues,
    /// The maximum number of iterations must be a positive value
    #[error("iterations must be positive")]
    InvalidIterations,
    /// The number of units in the spreading data must match the population size
    #[error("the number of units in the spreading data must match the population size")]
    InvalidSpreading,
    /// The bucket size must be positive
    #[error("bucket size must be positive")]
    InvalidBucketSize,
    /// The number of units in the balancing data must match the population size
    #[error("the number of units the balancing data must match the population size")]
    InvalidBalancing,
    /// Spreading data must be provided
    #[error("spreading data must be provided")]
    MissingSpreading,
    /// Balancing data must be provided
    #[error("balancing data must be provided")]
    MissingBalancing,
}

/// An alias for an `Result` returning a [`SamplingOptionsError`].
pub type SamplingOptionsResult<T> = Result<T, SamplingOptionsError>;
