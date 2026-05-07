// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
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

//! Horvitz-Thompson estimators (single count estimators)

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::PointSet;
use envisim_utils::kd_tree::searcher::KNearestNeighbourSearcher;
use envisim_utils::matrix::{
    Dimensions,
    MatrixBase,
    RawData,
};
use envisim_utils::probabilities::FloatProbabilities;
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
    SamplingOptionsError,
};
use num_traits::ToPrimitive;

/// Horvitz-Thompson estimator of a total
///
/// # Examples
/// ```
/// use envisim_estimate::horvitz_thompson::estimate;
///
/// let y = [0.0, 0.1, 0.2, 0.3, 0.4];
/// let pi = [0.2; 5];
///
/// estimate(&y, &pi).unwrap(); // Should be about 5.0
/// ```
pub fn estimate(y_values: &[f64], probabilities: &[f64]) -> Option<f64> {
    if y_values.len() != probabilities.len() {
        return None;
    }

    let mut est = 0.0;
    for (i, &y) in y_values.iter().enumerate() {
        let p = probabilities[i];
        if !FloatProbabilities::is_prob(p) {
            return None;
        } else if p == 0.0 {
            return Some(f64::NAN);
        }
        est += y / p;
    }
    Some(est)
}

/// Ratio estimator of total, using auxilliary variable `x_values`.
pub fn ratio(
    y_values: &[f64],
    x_values: &[f64],
    probabilities: &[f64],
    x_total: f64,
) -> Option<f64> {
    if !x_values.iter().all(|x| (0.0..).contains(x)) {
        return None;
    }
    Some(estimate(y_values, probabilities)? / estimate(x_values, probabilities)? * x_total)
}

/// Horvitz-Thompson estimator of variance of total estimate
pub fn variance<T>(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &MatrixBase<T>,
) -> Option<f64>
where
    T: RawData<Elem = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return None;
    } else if sample_size == 0 {
        return Some(0.0);
    }

    let yp = quotient(y_values, probabilities)?;

    // Do first unit first
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let p_i = probabilities[i];
        if yp[i].is_nan() {
            return Some(f64::NAN);
        }
        variance += yp[i].powi(2) * (1.0 - p_i);

        for j in 0..i {
            let p_ij = probabilities_second_order[(i, j)];
            if !FloatProbabilities::is_prob(p_ij) {
                return None;
            } else if p_ij == 0.0 {
                return Some(f64::NAN);
            }
            variance += 2.0 * yp[i] * yp[j] * (1.0 - p_i * probabilities[j] / p_ij);
        }
    }

    Some(variance)
}

/// Sen-Yates-Grundy estimator of variance of total estimate of fixed sized sample
pub fn syg_variance<T>(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &MatrixBase<T>,
) -> Option<f64>
where
    T: RawData<Elem = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return None;
    } else if sample_size == 0 {
        return Some(0.0);
    }

    let yp = quotient(y_values, probabilities)?;
    let mut variance: f64 = 0.0;

    for i in 1..sample_size {
        let p_i = probabilities[i];
        if yp[i].is_nan() {
            return Some(f64::NAN);
        }

        for j in 0..i {
            let p_ij = probabilities_second_order[(i, j)];
            if !FloatProbabilities::is_prob(p_ij) {
                return None;
            } else if p_ij == 0.0 {
                return Some(f64::NAN);
            }
            variance -= (yp[i] - yp[j]).powi(2) * (1.0 - p_i * probabilities[j] / p_ij);
        }
    }

    Some(variance)
}

/// Deville estimator of variance of total estimate
pub fn deville_variance(y_values: &[f64], probabilities: &[f64]) -> Option<f64> {
    let yp = quotient(y_values, probabilities)?;

    let q: Vec<f64> = probabilities.iter().map(|&p| 1.0 - p).collect();

    let s1mp = q.iter().sum::<f64>();
    let del = yp
        .iter()
        .zip(q.iter())
        .fold(0.0, |acc, (&a, &b)| acc + a * b);
    let s1mp_del = s1mp / del;
    let sak2 = q.iter().fold(0.0, |acc, &a| acc + a.powi(2)) / s1mp.powi(2);

    let dsum = yp
        .iter()
        .zip(q.iter())
        .fold(0.0, |acc, (&a, &b)| acc + (a - s1mp_del).powi(2) * b);

    Some(1.0 / (1.0 - sak2) * dsum)
}

/// Local mean estimator of variance of total estimate.
///
/// # References
/// Grafström, A., & Schelin, L. (2014).
/// How to select representative samples.
/// Scandinavian Journal of Statistics, 41(2), 277-290.
/// <https://doi.org/10.1111/sjos.12016>
pub fn local_mean_variance<PS, SOP, M>(
    y_values: &[f64],
    options: &SamplingOptions<PS, SOP, M>,
    n_neighbours: NonZeroUsize,
) -> Result<f64, SamplingOptionsError>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    let sample_size = y_values.len();

    if sample_size == 0 {
        return Ok(0.0);
    }

    let probabilities = options.probabilities().as_f64_slice();
    let tree = options.spreading()?.to_tree();
    // +1 since we search for self also
    let mut searcher =
        KNearestNeighbourSearcher::new(n_neighbours.checked_add(1).unwrap(), tree.data());

    let yp =
        quotient(y_values, probabilities.as_ref()).ok_or(SamplingOptionsError::InvalidSample)?;
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        if yp[i].is_nan() {
            return Ok(f64::NAN);
        }

        searcher
            .reset_from_slice(&tree.data().to_boxed_slice(i).unwrap())
            .unwrap()
            .search(&tree)
            .unwrap();
        let number_of_neighbours: f64 = searcher.neighbours().len().to_f64().unwrap();
        let local_mean: f64 = searcher
            .neighbours()
            .iter()
            .map(|n| yp[n.id()])
            .sum::<f64>()
            / number_of_neighbours;
        variance +=
            number_of_neighbours / (number_of_neighbours - 1.0) * (yp[i] - local_mean).powi(2);
    }

    Ok(variance)
}

fn quotient(ys: &[f64], ps: &[f64]) -> Option<Vec<f64>> {
    if ys.len() != ps.len() {
        return None;
    }

    let mut v = Vec::<f64>::with_capacity(ys.len());
    for (&y, &p) in ys.iter().zip(ps.iter()) {
        if !FloatProbabilities::is_prob(p) {
            return None;
        } else if p == 0.0 {
            v.push(f64::NAN);
        } else {
            v.push(y / p);
        }
    }
    Some(v)
}

#[cfg(test)]
mod test {
    use envisim_utils::test_utils::*;

    use super::*;

    const Y_VALS: [f64; 5] = [17.0, 130.0, 55.0, 80.0, 63.0];
    const PI_VALS: [f64; 5] = [0.0065, 0.0624, 0.0208, 0.0430, 0.0282];

    #[test]
    fn test_ht() {
        assert_delta!(estimate(&Y_VALS, &PI_VALS).unwrap(), 11437.46, 0.01);
    }
}
