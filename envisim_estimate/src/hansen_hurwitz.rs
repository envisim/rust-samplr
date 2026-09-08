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

//! Hansen-Hurwitz estimators (multiple count estimators)

use envisim_utils::matrix::{
    Dimensions,
    MatrixBase,
    SliceView,
};
use envisim_utils::utils::{
    ContiguousDataView,
    DataView,
    Number,
};

pub use crate::error::EstimationError;
use crate::error::EstimationResult;
use crate::utils::ymui_quotient;

/// Hansen-Hurwitz estimator of a total
///
/// # Examples
/// ```
/// # use envisim_estimate::hansen_hurwitz::*;
/// let y: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3, 0.4];
/// let mu: Vec<f64> = vec![0.2; 5];
/// let inc: Vec<f64> = vec![4.0, 3.0, 2.0, 1.0, 1.0];
/// estimate(&y, &mu, &inc)?; // Should be about 7.0
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the mus or incs are non-positive
#[inline]
pub fn estimate<Y, M, I>(y_values: Y, expected: M, inclusions: I) -> EstimationResult<f64>
where
    Y: DataView<Value: Number>,
    M: DataView<Id = Y::Id, Value = f64>,
    I: DataView<Id = Y::Id, Value: Number>,
{
    if y_values.len() != expected.len() || y_values.len() != inclusions.len() {
        return Err(EstimationError::InvalidSample);
    }

    expected
        .entries()
        .map(|(id, mu)| {
            let y = y_values
                .get(id)
                .ok_or(EstimationError::InvalidAuxiliaries)?;
            let inc = inclusions
                .get(id)
                .ok_or(EstimationError::InvalidNumberOfInclusions)?;
            ymui_quotient((*y, *mu, *inc))
        })
        .sum()
}

/// Hansen-Hurwitz estimator of variance of total estimate
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the mus or incs are non-positive
#[expect(clippy::missing_panics_doc, reason = "panic should be impossible")]
#[inline]
pub fn variance<Y, M1, I, M2>(
    y_values: Y,
    expected: M1,
    inclusions: I,
    expected_second_order: &MatrixBase<M2>,
) -> EstimationResult<f64>
where
    Y: ContiguousDataView<Value: Number>,
    M1: ContiguousDataView<Id = Y::Id, Value = f64>,
    I: ContiguousDataView<Id = Y::Id, Value: Number>,
    M2: SliceView<Id = Y::Id, Value = f64>,
{
    let sample_size = y_values.len();

    if sample_size != expected.len()
        || sample_size != inclusions.len()
        || sample_size != expected_second_order.nrow().get()
        || sample_size != expected_second_order.ncol().get()
    {
        return Err(EstimationError::InvalidSample);
    } else if sample_size == 0 {
        return Ok(0.0);
    }

    let ymui_box: Box<[f64]> = y_values
        .values()
        .copied()
        .zip(expected.values().copied())
        .zip(inclusions.values().copied())
        .map(|((y, m), i)| ymui_quotient((y, m, i)))
        .collect::<EstimationResult<Box<[f64]>>>()?;

    let mut variance: f64 = 0.0;

    for (i, &ymui_i) in ymui_box.iter().enumerate() {
        if ymui_i.is_nan() {
            return Ok(f64::NAN);
        }
        let mu_i = expected.get(i).expect("i to exist");

        let mu_ii = expected_second_order[(i, i)];
        if mu_ii < 0.0 || !mu_ii.is_finite() {
            return Err(EstimationError::InvalidExpectedNumberOfInclusions);
        }
        variance += ymui_i.powi(2) * (1.0 - mu_i.powi(2) / mu_ii);

        for j in 0..i {
            let ymui_j = ymui_box.get(j).expect("j to exist");
            let mu_j = expected.get(j).expect("j to exist");
            let mu_second_ord = expected_second_order[(i, j)];
            if mu_second_ord < 0.0 || !mu_second_ord.is_finite() {
                return Err(EstimationError::InvalidExpectedNumberOfInclusions);
            }
            variance += 2.0 * ymui_i * ymui_j * (1.0 - mu_i * mu_j / mu_second_ord);
        }
    }

    Ok(variance)
}

#[cfg(test)]
mod test {
    use super::*;

    const Y_VALS: [f64; 6] = [22.0, 30.0, 7.0, 25.0, 8.0, 12.0];
    const MU_VALS: [f64; 6] = [1.0, 0.5, 0.5, 0.2, 0.2, 0.6];

    #[test]
    fn test_hh() -> EstimationResult<()> {
        let indices: Vec<usize> = vec![1, 3, 5];
        let y: Vec<f64> = indices.entries().map(|&id| Y_VALS[id]).collect();
        let mu: Vec<f64> = indices.entries().map(|&id| MU_VALS[id]).collect();
        let inclusions: Vec<f64> = vec![1.0, 1.0, 1.0];
        assert_eq!(estimate(&y, &mu, &inclusions)?, 205.0);

        let indices: Vec<usize> = vec![0, 0, 2, 5];
        let mut indices_unique = indices.clone();
        indices_unique.dedup();
        let y: Vec<f64> = indices_unique.entries().map(|&id| Y_VALS[id]).collect();
        let mu: Vec<f64> = indices_unique.entries().map(|&id| MU_VALS[id]).collect();
        let inclusions: Vec<f64> = vec![2.0, 1.0, 1.0];
        assert_eq!(estimate(&y, &mu, &inclusions)?, 78.0);
        Ok(())
    }
}
