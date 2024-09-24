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

use std::num::NonZeroUsize;

#[non_exhaustive]
#[derive(Debug)]
pub enum InputError {
    // #[error("{0}")]
    // General(String),
    // 0 must be in closed range [1, 2]
    InvalidRangeF64(f64, f64, f64),
    InvalidRangeUsize(usize, usize, usize),
    // 0 cannot be 1
    InvalidValueF64(f64, f64),
    InvalidValueUsize(usize, usize),
    // 0 must be integer
    NotInteger(f64),
    // 0 must be 1
    InvalidSize(usize, usize),
    // empty slice
    IsEmpty,
    // slice contains duplicates
    NotUnique,
    // Missing input
    Missing(String),
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
    pub fn into_nonzero_usize(v: usize) -> Result<NonZeroUsize, InputError> {
        NonZeroUsize::new(v).ok_or(InputError::InvalidValueUsize(0, 0))
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

impl std::error::Error for InputError {}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // InputError::General(txt) => {
            //     write!(f, "{}", txt)
            // }
            InputError::InvalidRangeF64(x, a, b) => {
                write!(
                    f,
                    "invalid range: {x} must be in the closed range [{a}, {b}]"
                )
            }
            InputError::InvalidValueF64(x, c) => {
                write!(f, "invalid value: {x} cannot be {c}")
            }
            InputError::InvalidRangeUsize(x, a, b) => {
                write!(
                    f,
                    "invalid range: {x} must be in the closed range [{a}, {b}]"
                )
            }
            InputError::InvalidValueUsize(x, c) => {
                write!(f, "invalid value: {x} cannot be {c}")
            }
            InputError::NotInteger(x) => {
                write!(f, "value {x} must be integer")
            }
            InputError::InvalidSize(a, b) => {
                write!(f, "invalid size: {a} must be {b}")
            }
            InputError::IsEmpty => {
                write!(f, "slice is empty")
            }
            InputError::NotUnique => {
                write!(f, "slice contains duplicate elements")
            }
            InputError::Missing(ref txt) => {
                write!(f, "missing input {txt}")
            }
        }
    }
}
