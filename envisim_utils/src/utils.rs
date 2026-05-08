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

use num_traits::ToPrimitive;

/// Calculates the mean of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[expect(
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    reason = "usize to f64 conversion"
)]
#[must_use]
#[inline]
pub fn mean(vec: &[f64]) -> f64 { vec.iter().sum::<f64>() / vec.len().to_f64().unwrap() }

/// Calculates the variance of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[expect(
    clippy::missing_panics_doc,
    clippy::unwrap_used,
    reason = "usize to f64 conversion"
)]
#[must_use]
#[inline]
pub fn variance(vec: &[f64]) -> f64 {
    if vec.len() == 1 {
        return f64::NAN;
    }

    let mean = mean(vec);
    vec.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / (vec.len().to_f64().unwrap() - 1.0)
}

/// Calculates the standard deviance of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[must_use]
#[inline]
pub fn standard_deviance(vec: &[f64]) -> f64 { variance(vec).sqrt() }
