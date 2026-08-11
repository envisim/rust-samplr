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

use envisim_utils::sampling_options::SamplingOptionsError;
use thiserror::Error;

/// Sampling related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SamplingError {
    #[error("SamplingOptionsError: {0}")]
    Options(#[from] SamplingOptionsError),
    // max iterations reached
    #[error("max iterations ({0}) reached")]
    MaxIterations(NonZeroUsize),
    #[error("invalid stratification")]
    IncorrectStratification,
    #[error("draw probabilities must sum nominally to 1.0")]
    IncorrectDrawProbabilities,
    #[error("probabilities must sum to a nominal integer value")]
    IncorrectProbabilitiesIntegerSum,
    #[error("annealing temperature must be positive")]
    IncorrectAnnealingTemperature,
    #[error("annealing temperature cooling rate must be in (0.0, 1.0)")]
    IncorrectAnnealingRate,
    #[error("sample size must be positive")]
    ZeroSampleSize,
}
pub type SamplingResult<T> = Result<T, SamplingError>;
