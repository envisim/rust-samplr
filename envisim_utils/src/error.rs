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

//! General error types

use crate::indices::IndicesError;
use crate::kd_tree::{NodeError, SearcherError};
use thiserror::Error;

/// Input related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum InputError {
    #[error("{0}")]
    General(String),
    #[error("invalid range: {0} must be in the closed range [{1}, {2}]")]
    InvalidRangeF64(f64, f64, f64),
    #[error("invalid value: {0} cannot be {1}")]
    InvalidValueF64(f64, f64), // actual, invalid
    #[error("invalid range: {0} must be in the closed range [{1}, {2}]")]
    InvalidRangeUsize(usize, usize, usize),
    #[error("invalid value: {0} cannot be {1}")]
    InvalidValueUsize(usize, usize),
    #[error("value {0} must be integer")]
    NotInteger(f64),
    #[error("invalid size: {0} must be {1}")]
    InvalidSize(usize, usize),
    #[error("slice is empty")]
    IsEmpty,
    #[error("input does not contain unique elements")]
    NotUnique,
    #[error("missing input {0}")]
    Missing(String),
    #[error(transparent)]
    Node(#[from] NodeError),
}

impl InputError {
    #[inline]
    pub fn check_valid_f64(v: f64, invalid: f64) -> Result<(), InputError> {
        if v == invalid {
            return Err(InputError::InvalidValueF64(v, invalid));
        }
        Ok(())
    }
    #[inline]
    pub fn check_range_f64(v: f64, lo: f64, hi: f64) -> Result<(), InputError> {
        if v < lo || v > hi {
            return Err(InputError::InvalidRangeF64(v, lo, hi));
        }
        Ok(())
    }
    #[inline]
    pub fn check_positive(v: f64) -> Result<(), InputError> {
        if v == 0.0 {
            return Err(InputError::InvalidValueF64(v, 0.0));
        } else if v < 0.0 {
            return Err(InputError::InvalidRangeF64(v, 0.0, f64::INFINITY));
        }
        Ok(())
    }
    #[inline]
    pub fn check_valid_usize(v: usize, invalid: usize) -> Result<(), InputError> {
        if v == invalid {
            return Err(InputError::InvalidValueUsize(v, invalid));
        }
        Ok(())
    }
    #[inline]
    pub fn check_range_usize(v: usize, lo: usize, hi: usize) -> Result<(), InputError> {
        if v < lo || v > hi {
            return Err(InputError::InvalidRangeUsize(v, lo, hi));
        }

        Ok(())
    }
    #[inline]
    pub fn check_empty<T>(a: &[T]) -> Result<(), InputError> {
        if a.is_empty() {
            return Err(InputError::IsEmpty);
        }

        Ok(())
    }
    #[inline]
    pub fn check_lengths<TA, TB>(a: &[TA], b: &[TB]) -> Result<(), InputError> {
        InputError::check_sizes(a.len(), b.len())
    }
    #[inline]
    pub fn check_sizes(a: usize, b: usize) -> Result<(), InputError> {
        if a != b {
            return Err(InputError::InvalidSize(a, b));
        }

        Ok(())
    }
    #[inline]
    pub fn check_sample_size(sample_size: usize, population_size: usize) -> Result<(), InputError> {
        if population_size == 0 {
            return Err(InputError::InvalidValueUsize(population_size, 0));
        }
        Self::check_range_usize(sample_size, 0, population_size - 1)
    }
    #[inline]
    pub fn check_nan(value: f64) -> Result<(), InputError> {
        if value.is_nan() {
            return Err(InputError::InvalidValueF64(value, f64::NAN));
        }

        Ok(())
    }
    #[inline]
    pub fn check_integer(value: f64) -> Result<(), InputError> {
        if value != value.round() {
            return Err(InputError::NotInteger(value));
        }

        Ok(())
    }
    #[inline]
    pub fn check_integer_approx(value: f64, eps: f64) -> Result<(), InputError> {
        let round = value.round();
        if !(round - eps..round + eps).contains(&value) {
            return Err(InputError::NotInteger(value));
        }

        Ok(())
    }
    #[inline]
    pub fn check_integer_approx_equal(value: f64, target: f64, eps: f64) -> Result<(), InputError> {
        let round = value.round();
        if target != round {
            return Err(InputError::InvalidRangeF64(value, target, target));
        }
        Self::check_integer_approx(value, eps)
    }
}

/// Sampling related error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SamplingError {
    #[error("{0}")]
    General(String),
    #[error(transparent)]
    Indices(#[from] IndicesError),
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Node(#[from] NodeError),
    #[error(transparent)]
    Searcher(#[from] SearcherError),
    #[error("max iterations reached")]
    MaxIterations,
}
