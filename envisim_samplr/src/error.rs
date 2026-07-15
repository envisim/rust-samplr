// Copyright (C) 2025 Wilmer Prentius.
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

//! Errors for sampling algorithms

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::TreeError;
use envisim_utils::sampling_options::SamplingOptionsError;
use thiserror::Error;

/// Sampling related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SamplingError {
    /// Error derived from [`SamplingOptions`]
    #[error("SamplingOptionsError: {0}")]
    Options(#[from] SamplingOptionsError),
    /// Error derived from [`Tree`]
    #[error("TreeError: {0}")]
    Tree(#[from] TreeError),
    /// Max iterations reached
    #[error("max iterations ({0}) reached")]
    MaxIterations(NonZeroUsize),
    /// Invalid stratification
    #[error("invalid stratification")]
    IncorrectStratification,
    /// Draw probabilities must sum nominally to 1
    #[error("draw probabilities must sum nominally to 1.0")]
    IncorrectDrawProbabilities,
    /// Probabilities must sum to a nominal integer value
    #[error("probabilities must sum to a nominal integer value")]
    IncorrectProbabilitiesIntegerSum,
    /// Annealing temperatures must be positive
    #[error("annealing temperature must be positive")]
    IncorrectAnnealingTemperature,
    /// Anneling temperature cooling rate must be in (0.0, 1.0)
    #[error("annealing temperature cooling rate must be in (0.0, 1.0)")]
    IncorrectAnnealingRate,
    /// Sample size must be positive
    #[error("sample size must be positive")]
    ZeroSampleSize,
}

/// An alias for an `Result` returning a [`SamplingError`].
pub type SamplingResult<T> = Result<T, SamplingError>;
