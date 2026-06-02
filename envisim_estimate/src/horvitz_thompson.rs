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

//! Horvitz-Thompson estimators (single count estimators)

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::PointSet;
use envisim_utils::kd_tree::searcher::KNearestNeighbourSearcher;
use envisim_utils::matrix::{
    Dimensions,
    MatrixBase,
    RawData,
};
use envisim_utils::probabilities::Probability;
use envisim_utils::sampling_options::{
    ProbabilityOptions,
    SamplingOptions,
    SpreadingOptions,
};
use num_traits::ToPrimitive;

pub use crate::error::EstimationError;
use crate::error::EstimationResult;
use crate::utils::ypi_iter_to_vec;

/// Horvitz-Thompson estimator of a total
///
/// # Examples
/// ```
/// # use envisim_estimate::horvitz_thompson::*;
/// let y: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3, 0.4];
/// let pi: Vec<f64> = vec![0.2; 5];
/// estimate(&y, &pi).unwrap(); // Should be about 5.0
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid
#[inline]
pub fn estimate(y_values: &[f64], probabilities: &[f64]) -> EstimationResult<f64> {
    if y_values.len() != probabilities.len() {
        return Err(EstimationError::InvalidSample);
    }
    ypi_iter_to_vec(y_values.iter().zip(probabilities)).map(|q_vec| q_vec.iter().sum())
}

/// Ratio estimator of total, using auxilliary variable `x_values`.
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
/// Also returns an error if the xes are not positive.
#[inline]
pub fn ratio(
    y_values: &[f64],
    x_values: &[f64],
    probabilities: &[f64],
    x_total: f64,
) -> EstimationResult<f64> {
    if !x_values.iter().all(|x| (0.0..).contains(x)) {
        return Err(EstimationError::InvalidAuxiliaries);
    }
    let y_hat = estimate(y_values, probabilities)?;
    let x_hat = estimate(x_values, probabilities)?;

    Ok(y_hat / x_hat * x_total)
}

/// Horvitz-Thompson estimator of variance of total estimate
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[inline]
pub fn variance<T>(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &MatrixBase<T>,
) -> EstimationResult<f64>
where
    T: RawData<Elem = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return Err(EstimationError::InvalidSample);
    } else if sample_size == 0 {
        return Ok(0.0);
    }

    let yp = ypi_iter_to_vec(y_values.iter().zip(probabilities))?;

    // Do first unit first
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let p_i = probabilities[i];
        if yp[i].is_nan() {
            return Ok(f64::NAN);
        }
        variance += yp[i].powi(2) * (1.0 - p_i);

        for j in 0..i {
            let p_ij = probabilities_second_order[(i, j)];
            if !Probability::is_probability(p_ij, 1.0) {
                return Err(EstimationError::InvalidProbability);
            } else if p_ij == 0.0 {
                return Ok(f64::NAN);
            }
            variance += 2.0 * yp[i] * yp[j] * (1.0 - p_i * probabilities[j] / p_ij);
        }
    }

    Ok(variance)
}

/// Sen-Yates-Grundy estimator of variance of total estimate of fixed sized sample
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[inline]
pub fn syg_variance<T>(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &MatrixBase<T>,
) -> EstimationResult<f64>
where
    T: RawData<Elem = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return Err(EstimationError::InvalidSample);
    } else if sample_size == 0 {
        return Ok(0.0);
    }

    let yp = ypi_iter_to_vec(y_values.iter().zip(probabilities))?;
    let mut variance: f64 = 0.0;

    for i in 1..sample_size {
        let p_i = probabilities[i];
        if yp[i].is_nan() {
            return Ok(f64::NAN);
        }

        for j in 0..i {
            let p_ij = probabilities_second_order[(i, j)];
            if !Probability::is_probability(p_ij, 1.0) {
                return Err(EstimationError::InvalidProbability);
            } else if p_ij == 0.0 {
                return Ok(f64::NAN);
            }
            variance -= (yp[i] - yp[j]).powi(2) * (1.0 - p_i * probabilities[j] / p_ij);
        }
    }

    Ok(variance)
}

/// Deville estimator of variance of total estimate
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[inline]
pub fn deville_variance(y_values: &[f64], probabilities: &[f64]) -> EstimationResult<f64> {
    let yp = ypi_iter_to_vec(y_values.iter().zip(probabilities))?;

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

    Ok(1.0 / (1.0 - sak2) * dsum)
}

/// Local mean estimator of variance of total estimate.
///
/// # References
/// Grafström, A., & Schelin, L. (2014).
/// How to select representative samples.
/// Scandinavian Journal of Statistics, 41(2), 277-290.
/// <https://doi.org/10.1111/sjos.12016>
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
///
/// # Panics
/// Panics if `P` does not contains units `0..sample_size`.
#[inline]
pub fn local_mean_variance<PO, P, BAL>(
    y_values: &[f64],
    options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    n_neighbours: NonZeroUsize,
) -> EstimationResult<f64>
where
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize, Value = f64>,
{
    let sample_size = y_values.len();

    if sample_size == 0 {
        return Ok(0.0);
    }

    let probabilities = options.probabilities().to_slice_real();
    let tree = options.spreading().to_tree();
    // +1 since we search for self also
    let mut searcher = KNearestNeighbourSearcher::new(
        n_neighbours
            .checked_add(1)
            .expect("n_neibhours to be able to add 1"),
        tree.data(),
    );

    let yp = ypi_iter_to_vec(y_values.iter().zip(probabilities.iter()))?;
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        if yp[i].is_nan() {
            return Ok(f64::NAN);
        }

        searcher
            .reset_from_point(tree.data().get_coords(i).expect("i to exist in aux data"))
            .expect("tree data to be searchable")
            .search(&tree)
            .expect("search to be possible");
        let number_of_neighbours: f64 = searcher
            .neighbours()
            .len()
            .to_f64()
            .expect("neighbour len to convert to f64");
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

#[cfg(test)]
mod test {
    use envisim_utils::test_utils::*;

    use super::*;

    const Y_VALS: [f64; 5] = [17.0, 130.0, 55.0, 80.0, 63.0];
    const PI_VALS: [f64; 5] = [0.0065, 0.0624, 0.0208, 0.0430, 0.0282];

    #[test]
    fn test_ht() {
        assert_delta!(
            estimate(&Y_VALS, &PI_VALS).unwrap(),
            11437.46,
            Epsilon::new(0.01).unwrap()
        );
    }
}
