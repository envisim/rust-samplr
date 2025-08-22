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
        _ => panic!("usize ({}) too large to be converted to f64", v),
    }
}

/// Sums a vector of [`f64`]s.
/// Returns [`f64::NAN`]` if any element is `NaN`.
///
/// # Examples
/// ```
/// use envisim_utils::utils::sum;
/// assert_eq!(sum(&vec![0.0, 1.0, 2.0, 3.0]), 6.0);
/// assert!(sum(&vec![0.0, 1.0, 2.0, f64::NAN]).is_nan());
/// ```
#[inline]
pub fn sum(vec: &[f64]) -> f64 {
    vec.iter().fold(0.0, |acc, x| acc + x)
}

/// Calculates the mean of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`]` if any element is `NaN`.
#[inline]
pub fn mean(vec: &[f64]) -> f64 {
    sum(vec) / usize_to_f64(vec.len())
}

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
pub fn standard_deviance(vec: &[f64]) -> f64 {
    variance(vec).sqrt()
}
