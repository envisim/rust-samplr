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

//! Functions for calculating probabilities proportional to size

use crate::utils::usize_to_f64;
use crate::{InputError, Probabilities};

/// Draw probabilities proportional to size.
/// Given an array of positive values, returns draw probabilities proportional to size.
/// Returns an error if any value is non-positive.
pub fn pps_from_slice(arr: &[f64]) -> Result<Probabilities, InputError> {
    if arr.is_empty() {
        return Probabilities::new(0, 0.0);
    }

    let mut sum: f64 = 0.0;

    for &x in arr {
        InputError::check_range_f64(x, 0.0, f64::INFINITY)
            .and(InputError::check_valid_f64(x, 0.0))?;
        sum += x;
    }

    Probabilities::with_values(&arr.iter().map(|&x| x / sum).collect::<Vec<f64>>())
}

/// Inclusion probabilities proportional to size (approximate).
/// Given an array of positive values, returns the inclusion probabilities proportional to size.
/// Returns an error if any value is non-positive.
///
/// The caluclations are done by iteratively rescaling the inclusion probabilities.
pub fn pips_from_slice(arr: &[f64], sample_size: usize) -> Result<Probabilities, InputError> {
    if arr.is_empty() {
        return Probabilities::new(0, 0.0);
    }

    if arr.len() < sample_size {
        return Probabilities::new(arr.len(), 1.0);
    }

    arr.iter().try_for_each(|&x| {
        InputError::check_range_f64(x, 0.0, f64::INFINITY).and(InputError::check_valid_f64(x, 0.0))
    })?;

    let mut n = usize_to_f64(sample_size);

    let mut pips = Probabilities::new(arr.len(), 0.0)?;
    let mut failed: bool = true;

    while failed && n > 0.0 {
        failed = false;
        let sum: f64 = arr
            .iter()
            .enumerate()
            .filter(|(i, _)| pips[*i] < 1.0)
            .fold(0.0, |acc, (_, &x)| acc + x);
        let curr_n = n;

        arr.iter().enumerate().for_each(|(i, &x)| {
            if pips[i] >= 1.0 {
                return;
            }

            let p = (x * curr_n) / sum;
            pips[i] = p.min(1.0);

            if p >= 1.0 {
                n -= 1.0;

                if !failed && p > 1.0 {
                    failed = true;
                }
            }
        });
    }

    Ok(pips)
}
