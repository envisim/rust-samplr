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

use envisim_utils::{kd_tree::NodeError, IndicesError, InputError};
use std::num::NonZeroUsize;

/// Sampling related error types
#[non_exhaustive]
#[derive(Debug)]
pub enum SamplingError {
    Indices(IndicesError),
    Input(InputError),
    Node(NodeError),
    // max iterations reached
    MaxIterations(NonZeroUsize),
}

impl std::error::Error for SamplingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            SamplingError::Indices(ref err) => Some(err),
            SamplingError::Input(ref err) => Some(err),
            SamplingError::Node(ref err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            SamplingError::Indices(ref err) => err.fmt(f),
            SamplingError::Input(ref err) => err.fmt(f),
            SamplingError::Node(ref err) => err.fmt(f),
            SamplingError::MaxIterations(max_iter) => {
                write!(f, "max iterations ({max_iter}) reached")
            }
        }
    }
}

impl From<IndicesError> for SamplingError {
    fn from(err: IndicesError) -> SamplingError {
        SamplingError::Indices(err)
    }
}
impl From<InputError> for SamplingError {
    fn from(err: InputError) -> SamplingError {
        SamplingError::Input(err)
    }
}
impl From<NodeError> for SamplingError {
    fn from(err: NodeError) -> SamplingError {
        SamplingError::Node(err)
    }
}
