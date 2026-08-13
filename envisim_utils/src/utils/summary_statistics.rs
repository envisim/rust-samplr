// Copyright (C) 2026 Wilmer Prentius.
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

//! Summary statistics

use num_traits::ToPrimitive;

/// Calculates the mean of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[expect(clippy::missing_panics_doc, reason = "usize to f64")]
#[must_use]
#[inline]
pub fn mean<'bitem, I>(iter: I) -> f64
where
    I: ExactSizeIterator<Item = &'bitem f64>,
{
    let len = iter.len().to_f64().expect("usize -> f64");
    let sum: f64 = iter.sum();
    sum / len
}

/// Calculates the variance of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[expect(clippy::missing_panics_doc, reason = "usize to f64")]
#[must_use]
#[inline]
pub fn variance<'bitem, I>(iter: I) -> f64
where
    I: ExactSizeIterator<Item = &'bitem f64> + Clone,
{
    let len = iter.len();
    if len == 1 {
        f64::NAN
    } else {
        let mean = mean(iter.clone());
        let sum: f64 = iter.map(|v| (v - mean).powi(2)).sum();
        sum / (len - 1).to_f64().expect("usize -> f64")
    }
}

/// Calculates the standard deviation of a vector.
/// Panics if the length of the vector is larger than [`u32::MAX`].
/// Returns [`f64::NAN`] if any element is `NaN`.
#[must_use]
#[inline]
pub fn sd<'bitem, I>(iter: I) -> f64
where
    I: ExactSizeIterator<Item = &'bitem f64> + Clone,
{
    variance(iter).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_mean() {
        assert_delta!(mean(Data10::PROB_U.iter()), 0.5);
    }

    #[test]
    fn test_variance() {
        assert_delta!(
            variance(Data10::PROB_U.iter()),
            0.045556,
            Epsilon::new(1e-5).unwrap()
        );
    }

    #[test]
    fn test_sd() {
        assert_delta!(
            sd(Data10::PROB_U.iter()),
            0.2134375,
            Epsilon::new(1e-5).unwrap()
        );
    }
}
