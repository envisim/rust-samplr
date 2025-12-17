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

use std::num::NonZeroUsize;

use envisim_utils::error::InputError;
use envisim_utils::sampling_options::SamplingOptionsError;

/// Sampling related error types
#[non_exhaustive]
#[derive(Debug)]
pub enum SamplingError {
    Options(SamplingOptionsError),
    Input(InputError),
    // max iterations reached
    MaxIterations(NonZeroUsize),
    IncorrectStratification,
}

impl std::error::Error for SamplingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            SamplingError::Input(ref err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use SamplingError::*;
        match *self {
            Options(ref err) => err.fmt(f),
            Input(ref err) => err.fmt(f),
            MaxIterations(max_iter) => {
                write!(f, "max iterations ({max_iter}) reached")
            }
            IncorrectStratification => {
                write!(f, "incorrect stratification")
            }
        }
    }
}

impl From<InputError> for SamplingError {
    fn from(err: InputError) -> SamplingError { SamplingError::Input(err) }
}

impl From<SamplingOptionsError> for SamplingError {
    fn from(err: SamplingOptionsError) -> SamplingError { SamplingError::Options(err) }
}
