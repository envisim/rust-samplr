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

use std::iter::repeat_n;

use num_traits::ToPrimitive;

pub use self::error::PipsError;
use crate::probabilities::{
    Probability,
    ProbabilitySet,
    RealProbabilityValue,
};
use crate::sampling_options::Epsilon;

/// Draw probabilities proportional to size.
/// Given an array of positive values, returns draw probabilities proportional to size.
///
/// # Errors
/// Returns an error if any value is non-positive.
#[expect(clippy::missing_panics_doc, reason = "impossible")]
#[inline]
pub fn pps_from_slice(arr: &[f64]) -> Result<ProbabilitySet<f64>, PipsError> {
    if arr.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    }

    let mut sum: f64 = 0.0;

    for x in arr {
        // x is finite and positive
        if !x.is_finite() || (..=0.0).contains(x) {
            return Err(PipsError::InvalidAuxiliary);
        }
        sum += *x;
    }

    Ok(
        ProbabilitySet::<f64>::try_new(arr.iter().map(|&x| x / sum), 1.0, Epsilon::default())
            .expect("1.0 > 0"),
    )
}

/// Inclusion probabilities proportional to size (approximate).
/// Given an array of positive values, returns the inclusion probabilities proportional to size.
///
/// The caluclations are done by iteratively rescaling the inclusion probabilities.
///
/// # Errors
/// Returns an error if any value is non-positive.
#[expect(clippy::missing_panics_doc, reason = "usize to f64 conversion")]
#[inline]
pub fn pips_from_slice(arr: &[f64], sample_size: usize) -> Result<ProbabilitySet<f64>, PipsError> {
    if arr.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    }

    if arr.len() < sample_size {
        return Ok(ProbabilitySet::<f64>::try_new(
            repeat_n(1.0, arr.len()),
            1.0,
            Epsilon::default(),
        )
        .expect("1.0 > 0"));
    }

    if arr.iter().any(|x| !x.is_normal() || (..0.0).contains(x)) {
        return Err(PipsError::InvalidAuxiliary);
    }

    let mut n = sample_size.to_f64().expect("usize to f64 conversion");

    let mut pips =
        ProbabilitySet::<f64>::try_new(repeat_n(0.0, arr.len()), 1.0, Epsilon::default())
            .expect("1.0 > 0");
    let mut failed = true;

    while failed && n > 0.0 {
        failed = false;
        let sum: f64 = arr
            .iter()
            .enumerate()
            .filter(|(i, _)| !pips[*i].is_full())
            .fold(0.0, |acc, (_, &x)| acc + x);
        let curr_n = n;

        arr.iter().enumerate().for_each(|(i, &x)| {
            if pips[i].is_full() {
                return;
            }

            let p = (x * curr_n) / sum;
            pips.set(
                i,
                Probability::new_real(p.min(1.0), Epsilon::default()).expect("p to be contained"),
            );

            if pips[i].is_full() {
                n -= 1.0;

                if p > 1.0 {
                    failed = true;
                }
            }
        });
    }

    Ok(pips)
}

mod error {
    //! Pips errors

    #[non_exhaustive]
    #[derive(Debug)]
    pub enum PipsError {
        InvalidAuxiliary,
        NoAuxiliaries,
    }
    #[expect(clippy::absolute_paths, reason = "possible override")]
    impl std::error::Error for PipsError {}
    #[expect(clippy::absolute_paths, reason = "possible override")]
    impl std::fmt::Display for PipsError {
        #[expect(clippy::enum_glob_use, reason = "handy to use in a match")]
        #[inline]
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            use PipsError::*;
            match *self {
                InvalidAuxiliary => write!(f, "auxiliaries must be positive"),
                NoAuxiliaries => write!(f, "slice contains no auxiliaries"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn pps() {
        let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
        let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];

        let pps = pps_from_slice(&dt1).unwrap();
        assert_vec!(pps.to_raw(), [0.1, 0.2, 0.3, 0.4]);

        assert!(pps_from_slice(&dt2).is_err());
    }

    #[test]
    fn pips() {
        let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
        let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];
        let dt3 = vec![1.0f64, 1.0, 1.0, 7.0];

        let pips1 = pips_from_slice(&dt1, 2).unwrap();
        assert_vec!(pips1.to_raw(), [0.2, 0.4, 0.6, 0.8]);

        assert!(pips_from_slice(&dt2, 2).is_err());

        let pips3 = pips_from_slice(&dt3, 2).unwrap();
        assert_vec!(pips3.to_raw(), [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
    }
}
