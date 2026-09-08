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
use envisim_utils::kd_tree::searcher::{
    KNearestNeighbourSearcher,
    NeighbourView,
};
use envisim_utils::matrix::{
    Dimensions,
    MatrixBase,
    SliceView,
};
use envisim_utils::probabilities::{
    Probability,
    ProbabilityValue,
};
use envisim_utils::sampling_options::{
    ProbabilitiesSpec,
    SamplingOptions,
    SpreadingOptions,
};
use envisim_utils::utils::{
    ConstructableDataView,
    ContiguousDataView,
    DataView,
    Number,
};
use num_traits::ToPrimitive;

pub use crate::error::EstimationError;
use crate::error::EstimationResult;
use crate::utils::ypi_quotient;

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
pub fn estimate<Y, P>(y_values: Y, probabilities: P) -> EstimationResult<f64>
where
    Y: DataView<Value: Number>,
    P: DataView<Id = Y::Id, Value = f64>,
{
    if y_values.len() != probabilities.len() {
        return Err(EstimationError::InvalidSample);
    }

    probabilities
        .entries()
        .map(|(id, p)| {
            let y = y_values
                .get(id)
                .ok_or(EstimationError::InvalidAuxiliaries)?;
            ypi_quotient((*y, *p))
        })
        .sum()
}

/// Ratio estimator of total, using auxilliary variable `x_values`.
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
/// Also returns an error if the xes are not positive.
#[inline]
pub fn ratio<Y, X, P>(
    y_values: Y,
    x_values: X,
    probabilities: P,
    x_total: X::Value,
) -> EstimationResult<f64>
where
    Y: DataView<Value: Number>,
    X: DataView<Id = Y::Id, Value: Number>,
    P: DataView<Id = Y::Id, Value = f64>,
{
    if y_values.len() != probabilities.len() || y_values.len() != x_values.len() {
        return Err(EstimationError::InvalidSample);
    }

    let mut y_hat = 0.0;
    let mut x_hat = 0.0;

    for (id, p) in probabilities.entries() {
        let y = y_values
            .get(id)
            .ok_or(EstimationError::InvalidAuxiliaries)?;
        let x = x_values
            .get(id)
            .ok_or(EstimationError::InvalidAuxiliaries)?;

        y_hat += ypi_quotient((*y, *p))?;
        x_hat += ypi_quotient((*x, *p))?;
    }

    x_total
        .to_f64()
        .map(|x| y_hat / x_hat * x)
        .ok_or(EstimationError::InvalidAuxiliaries)
}

/// Horvitz-Thompson estimator of variance of total estimate
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[expect(clippy::missing_panics_doc, reason = "panic should be impossible")]
#[inline]
pub fn variance<Y, P1, P2>(
    y_values: Y,
    probabilities: P1,
    probabilities_second_order: &MatrixBase<P2>,
) -> EstimationResult<f64>
where
    Y: ContiguousDataView<Value: Number>,
    P1: ContiguousDataView<Id = Y::Id, Value = f64>,
    P2: SliceView<Id = Y::Id, Value = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities.len()
        || sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return Err(EstimationError::InvalidSample);
    } else if sample_size == 0 {
        return Ok(0.0);
    }

    let yp_box: Box<[f64]> = y_values
        .values()
        .copied()
        .zip(probabilities.values().copied())
        .map(ypi_quotient)
        .collect::<EstimationResult<Box<[f64]>>>()?;

    // Do first unit first
    let mut variance: f64 = 0.0;

    for (i, &yp_i) in yp_box.iter().enumerate() {
        if yp_i.is_nan() {
            return Ok(f64::NAN);
        }
        let p_i = probabilities.get(i).expect("i to exist");
        variance += yp_box[i].powi(2) * (1.0 - p_i);

        for j in 0..i {
            let yp_j = yp_box.get(j).expect("j to exist");
            let p_j = probabilities.get(j).expect("j to exist");
            let p_second_ord = probabilities_second_order[(i, j)];
            if !Probability::<f64>::is_probability(p_second_ord, 1.0) {
                return Err(EstimationError::InvalidProbability);
            } else if p_second_ord == 0.0 {
                return Ok(f64::NAN);
            }
            variance += 2.0 * yp_i * yp_j * (1.0 - p_i * p_j / p_second_ord);
        }
    }

    Ok(variance)
}

/// Sen-Yates-Grundy estimator of variance of total estimate of fixed sized sample
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[expect(clippy::missing_panics_doc, reason = "panic should be impossible")]
#[inline]
pub fn syg_variance<Y, P1, P2>(
    y_values: Y,
    probabilities: P1,
    probabilities_second_order: &MatrixBase<P2>,
) -> EstimationResult<f64>
where
    Y: ContiguousDataView<Value: Number>,
    P1: ContiguousDataView<Value = f64>,
    P2: SliceView<Value = f64>,
{
    let sample_size = y_values.len();

    if sample_size != probabilities_second_order.nrow().get()
        || sample_size != probabilities_second_order.ncol().get()
    {
        return Err(EstimationError::InvalidSample);
    } else if sample_size == 0 {
        return Ok(0.0);
    }

    let yp: Box<[f64]> = y_values
        .values()
        .copied()
        .zip(probabilities.values().copied())
        .map(ypi_quotient)
        .collect::<EstimationResult<Box<[f64]>>>()?;
    let mut variance: f64 = 0.0;

    for i in 1..sample_size {
        let p_i = probabilities.get(i).expect("i to exist");
        if yp[i].is_nan() {
            return Ok(f64::NAN);
        }

        for j in 0..i {
            let p_j = probabilities.get(j).expect("j to exist");
            let p_second_ord = probabilities_second_order[(i, j)];
            if !Probability::is_probability(p_second_ord, 1.0) {
                return Err(EstimationError::InvalidProbability);
            } else if p_second_ord == 0.0 {
                return Ok(f64::NAN);
            }
            variance -= (yp[i] - yp[j]).powi(2) * (1.0 - p_i * p_j / p_second_ord);
        }
    }

    Ok(variance)
}

/// Deville estimator of variance of total estimate
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[inline]
pub fn deville_variance<Y, P>(y_values: Y, probabilities: P) -> EstimationResult<f64>
where
    Y: ContiguousDataView<Value: Number>,
    P: ContiguousDataView<Id = Y::Id, Value = f64>,
{
    let yp: Box<[f64]> = y_values
        .values()
        .copied()
        .zip(probabilities.values().copied())
        .map(ypi_quotient)
        .collect::<EstimationResult<Box<[f64]>>>()?;

    let q: Box<[f64]> = probabilities.values().map(|&p| 1.0 - p).collect();

    let s1mp: f64 = q.values().sum();
    let del: f64 = yp.values().zip(q.values()).map(|(&a, &b)| a * b).sum();
    let s1mp_del = s1mp / del;
    let sak2 = q.values().map(|&a| a.powi(2)).sum::<f64>() / s1mp.powi(2);

    let dsum: f64 = yp
        .values()
        .zip(q.values())
        .map(|(&a, &b)| (a - s1mp_del).powi(2) * b)
        .sum();

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
pub fn local_mean_variance<Y, PO, P, BAL>(
    y_values: Y,
    options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    n_neighbours: NonZeroUsize,
) -> EstimationResult<f64>
where
    Y: ConstructableDataView<Value: Number>,
    PO: ProbabilitiesSpec<Id = Y::Id, Real = f64>,
    P: PointSet<Id = Y::Id, Value = f64>,
{
    let sample_size = y_values.len();
    if sample_size == 0 {
        return Ok(0.0);
    }

    let tree = options.spreading().to_tree();
    // +1 since we search for self also
    let mut searcher = KNearestNeighbourSearcher::new(
        n_neighbours
            .checked_add(1)
            .expect("n_neibhours to be able to add 1"),
        tree.data(),
    );

    let yp_box = Y::try_from_iter(y_values.entries().map(|(id, y)| {
        let y = y.to_f64().ok_or(EstimationError::InvalidAuxiliaries)?;
        let p = options
            .probabilities()
            .get_real(id)
            .ok_or(EstimationError::InvalidProbability)?;
        Result::<(Y::Id, f64), EstimationError>::Ok((id, y / p))
    }))?;

    let mut variance: f64 = 0.0;

    for (id, yp) in yp_box.entries() {
        if yp.is_nan() {
            return Ok(f64::NAN);
        }

        searcher
            .reset_from_point(tree.data().coords(id).expect("i to exist in aux data"))
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
            .map(|n| *yp_box.get(*n.id()).expect("neighbour to exist"))
            .sum::<f64>()
            / number_of_neighbours;

        variance +=
            number_of_neighbours / (number_of_neighbours - 1.0) * (*yp - local_mean).powi(2);
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
