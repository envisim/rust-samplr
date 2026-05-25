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

/// Sampling related error types
#[non_exhaustive]
#[derive(Debug)]
pub enum EstimationError {
    Options(SamplingOptionsError),
    InvalidSample,
    InvalidProbability,
    InvalidExpectedNumberOfInclusions,
    InvalidNumberOfInclusions,
    InvalidAuxiliaries,
}

pub type EstimationResult<T> = Result<T, EstimationError>;

#[expect(clippy::absolute_paths, reason = "possible override")]
impl std::error::Error for EstimationError {
    #[inline]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        #[expect(clippy::wildcard_enum_match_arm, reason = "Options is a special case")]
        match self {
            EstimationError::Options(err) => Some(err),
            _ => None,
        }
    }
}

#[expect(clippy::absolute_paths, reason = "possible override")]
impl std::fmt::Display for EstimationError {
    #[expect(clippy::enum_glob_use, reason = "handy to use in a match")]
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use EstimationError::*;
        match self {
            Options(err) => err.fmt(f),
            InvalidSample => write!(f, "sample is invalid, either empty or contains oob units"),
            InvalidProbability => write!(f, "probability is not in [0.0, 1.0]"),
            InvalidExpectedNumberOfInclusions => {
                write!(f, "expected number of inclusions is not non-negative")
            }
            InvalidNumberOfInclusions => {
                write!(f, "number of inclusions is not non-negative")
            }
            InvalidAuxiliaries => write!(f, "auxiliaries are not valid"),
        }
    }
}

impl From<SamplingOptionsError> for EstimationError {
    #[inline]
    fn from(err: SamplingOptionsError) -> Self { EstimationError::Options(err) }
}
