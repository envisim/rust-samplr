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

//! Small utility functions

/// Converts a [`usize´] to a [`f64`], by first trying to convert it to a [`u32`]. Panics otherwise.
///
/// # Examples
/// ```
/// use envisim_utils::utils::usize_to_f64;
/// assert_eq!(usize_to_f64(140usize), 140.0f64);
/// assert_eq!(usize_to_f64(4_294_967_295usize), 4_294_967_295.0f64);
/// ```
#[inline]
pub fn usize_to_f64(v: usize) -> f64 {
    match u32::try_from(v) {
        Ok(v) => f64::from(v),
        _ => panic!("usize ({v}) too large to be converted to f64"),
    }
}

/// Converts a [`f64´] to a [`usize`], by first trying to vonvert it to a [`u32`]. Rounds the float.
/// Panics on fail.
///
/// # Examples
/// ```
/// use envisim_utils::utils::f64_to_usize;
/// assert_eq!(f64_to_usize(140.1f64), 140usize);
/// assert_eq!(f64_to_usize(4_294_967_295.0f64), 4_294_967_295usize);
/// ```
#[inline]
pub fn f64_to_usize(f: f64) -> usize {
    const U32_MAX: f64 = 4_294_967_295.0f64;
    if f < 0.0 {
        panic!("f ({f}) is negative and cannot be converted to f64");
    } else if f > U32_MAX {
        panic!("f ({f}) too large to be converted to f64")
    }
    f.round() as usize
}

/// Calculates the mean of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`]` if any element is `NaN`.
#[inline]
pub fn mean(vec: &[f64]) -> f64 { vec.iter().sum::<f64>() / usize_to_f64(vec.len()) }

/// Calculates the variance of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`]` if any element is `NaN`.
#[inline]
pub fn variance(vec: &[f64]) -> f64 {
    if vec.len() == 1 {
        return f64::NAN;
    }

    let mean = mean(vec);
    vec.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / usize_to_f64(vec.len() - 1)
}

/// Calculates the standard deviance of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`]` if any element is `NaN`.
#[inline]
pub fn standard_deviance(vec: &[f64]) -> f64 { variance(vec).sqrt() }
