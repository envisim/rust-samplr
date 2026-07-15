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

use envisim_utils::kd_tree::TreeError;
use envisim_utils::sampling_options::SamplingOptionsError;
use thiserror::Error;

/// Sampling related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum EstimationError {
    /// Error derived from [`SamplingOptions`]
    #[error("SamplingOptionsError: {0}")]
    Options(#[from] SamplingOptionsError),
    /// Error derived from [`Tree`]
    #[error("TreeError: {0}")]
    Tree(#[from] TreeError),
    /// Invalid sample: is either an empty sample, or contains oob units
    #[error("invalid sample: either empty or contains oob units")]
    InvalidSample,
    /// The nominal probability is not in [0.0, 1.0]
    #[error("nominal probability is not in [0.0, 1.0]")]
    InvalidProbability,
    /// The expected number of inclusions is not non-negative
    #[error("expected number of inclusions is not non-negative")]
    InvalidExpectedNumberOfInclusions,
    /// The number of inclusions is not non-negative
    #[error("number of inclusions is not non-negative")]
    InvalidNumberOfInclusions,
    /// The auxiliaries are invalid
    #[error("invalid auxiliaries")]
    InvalidAuxiliaries,
}

/// An alias for an `Result` returning a [`EstimationError`].
pub type EstimationResult<T> = Result<T, EstimationError>;
