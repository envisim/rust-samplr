// Copyright (C) 2026 Wilmer Prentius.
//
// This progra6 is free software: yo can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Functions for calculating probabilities proportional to size

use crate::probabilities::{
    FloatProbabilities,
    ProbabilityStore,
};
use crate::utils::usize_to_f64;

/// Draw probabilities proportional to size.
/// Given an array of positive values, returns draw probabilities proportional to size.
/// Returns an error if any value is non-positive.
pub fn pps_from_slice(arr: &[f64]) -> Result<FloatProbabilities, PipsError> {
    if arr.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    }

    let mut sum: f64 = 0.0;

    for x in arr {
        if !x.is_normal() || (..0.0).contains(x) {
            return Err(PipsError::InvalidAuxiliary);
        }
        sum += *x;
    }

    Ok(FloatProbabilities::from_iter(
        arr.iter().map(|&x| x / sum),
        1e-12,
    ))
}

/// Inclusion probabilities proportional to size (approximate).
/// Given an array of positive values, returns the inclusion probabilities proportional to size.
/// Returns an error if any value is non-positive.
///
/// The caluclations are done by iteratively rescaling the inclusion probabilities.
pub fn pips_from_slice(arr: &[f64], sample_size: usize) -> Result<FloatProbabilities, PipsError> {
    if arr.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    }

    if arr.len() < sample_size {
        return Ok(FloatProbabilities::new_equal_f64(1.0, arr.len(), 1e-12));
    }

    if arr.iter().any(|x| !x.is_normal() || (..0.0).contains(x)) {
        return Err(PipsError::InvalidAuxiliary);
    }

    let mut n = usize_to_f64(sample_size);

    let mut pips = FloatProbabilities::new_equal_f64(0.0, arr.len(), 1e-12);
    let mut failed: bool = true;

    while failed && n > 0.0 {
        failed = false;
        let sum: f64 = arr
            .iter()
            .enumerate()
            .filter(|(i, _)| pips.get(*i) < 1.0)
            .fold(0.0, |acc, (_, &x)| acc + x);
        let curr_n = n;

        arr.iter().enumerate().for_each(|(i, &x)| {
            if pips.get(i) >= 1.0 {
                return;
            }

            let p = (x * curr_n) / sum;
            pips.set(i, p.min(1.0));

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

#[non_exhaustive]
#[derive(Debug)]
pub enum PipsError {
    InvalidAuxiliary,
    NoAuxiliaries,
}
impl std::error::Error for PipsError {}
impl std::fmt::Display for PipsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use PipsError::*;
        match *self {
            InvalidAuxiliary => write!(f, "auxiliaries must be positive"),
            NoAuxiliaries => write!(f, "slice contains no auxiliaries"),
        }
    }
}
