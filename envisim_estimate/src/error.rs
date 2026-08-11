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

//! Estimation errors

use envisim_utils::sampling_options::SamplingOptionsError;
use thiserror::Error;

/// Sampling related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum EstimationError {
    #[error("SamplingOptionsError: {0}")]
    Options(#[from] SamplingOptionsError),
    #[error("invalid sample: either empty or contains oob units")]
    InvalidSample,
    #[error("nominal probability is not in [0.0, 1.0]")]
    InvalidProbability,
    #[error("expected number of inclusions is not non-negative")]
    InvalidExpectedNumberOfInclusions,
    #[error("number of inclusions is not non-negative")]
    InvalidNumberOfInclusions,
    #[error("invalid auxiliaries")]
    InvalidAuxiliaries,
}

pub type EstimationResult<T> = Result<T, EstimationError>;
