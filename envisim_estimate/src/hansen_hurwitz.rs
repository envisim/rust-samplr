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

//! Hansen-Hurwitz estimators (multiple count estimators)

use envisim_utils::matrix::{
    MatrixBase,
    RawData,
};

fn quotient(ys: &[f64], ms: &[f64], incs: &[f64]) -> Option<Vec<f64>> {
    let sample_size = ys.len();
    if sample_size != ms.len() || sample_size != incs.len() {
        return None;
    }

    let mut v = Vec::<f64>::with_capacity(ys.len());
    for i in 0..sample_size {
        let y = ys[i];
        let m = ms[i];
        let inc = incs[i];

        if m < 0.0 || inc < 0.0 {
            return None;
        } else if m == 0.0 {
            v.push(f64::NAN);
        } else {
            v.push(y / m * inc);
        }
    }
    Some(v)
}

/// Hansen-Hurwitz estimator of a total
///
/// # Examples
/// ```
/// use envisim_estimate::hansen_hurwitz::estimate;
///
/// let y = [0.0, 0.1, 0.2, 0.3, 0.4];
/// let mu = [0.2; 5];
/// let inc = [4.0, 3.0, 2.0, 1.0, 1.0];
///
/// estimate(&y, &mu, &inc).unwrap(); // Should be about 7.0
/// ```
pub fn estimate(y_values: &[f64], expected: &[f64], inclusions: &[f64]) -> Option<f64> {
    quotient(y_values, expected, inclusions).map(|q| q.iter().sum::<f64>())
}

/// Hansen-Hurwitz estimator of variance of total estimate
pub fn variance<T>(
    y_values: &[f64],
    expected: &[f64],
    inclusions: &[f64],
    expected_second_order: &MatrixBase<T>,
) -> Option<f64>
where
    T: RawData<Elem = f64>,
{
    let sample_size = y_values.len();

    if sample_size != expected_second_order.nrow().get()
        || sample_size != expected_second_order.ncol().get()
    {
        return None;
    } else if sample_size == 0 {
        return Some(0.0);
    }

    let ypi = quotient(y_values, expected, inclusions)?;
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        if ypi[i].is_nan() {
            return Some(f64::NAN);
        }
        variance += ypi[i].powi(2) * (1.0 - expected[i].powi(2) / expected_second_order[(i, i)]);

        for j in 0..i {
            variance += 2.0
                * ypi[i]
                * ypi[j]
                * (1.0 - expected[i] * expected[j] / expected_second_order[(i, j)]);
        }
    }

    Some(variance)
}

#[cfg(test)]
mod test {
    use super::*;

    const Y_VALS: [f64; 6] = [22.0, 30.0, 7.0, 25.0, 8.0, 12.0];
    const MU_VALS: [f64; 6] = [1.0, 0.5, 0.5, 0.2, 0.2, 0.6];

    #[test]
    fn test_hh() {
        let indices: Vec<usize> = vec![1, 3, 5];
        let y: Vec<f64> = indices.iter().map(|&id| Y_VALS[id]).collect();
        let mu: Vec<f64> = indices.iter().map(|&id| MU_VALS[id]).collect();
        let inclusions: Vec<f64> = vec![1.0, 1.0, 1.0];
        assert_eq!(estimate(&y, &mu, &inclusions), Some(205.0));

        let indices: Vec<usize> = vec![0, 0, 2, 5];
        let mut indices_unique = indices.clone();
        indices_unique.dedup();
        let y: Vec<f64> = indices_unique.iter().map(|&id| Y_VALS[id]).collect();
        let mu: Vec<f64> = indices_unique.iter().map(|&id| MU_VALS[id]).collect();
        let inclusions: Vec<f64> = vec![2.0, 1.0, 1.0];
        assert_eq!(estimate(&y, &mu, &inclusions), Some(78.0));
    }
}
