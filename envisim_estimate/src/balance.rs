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

//! Balance deviation

pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    BalancingOptions,
    ProbabilitiesSpec,
    SpreadingOptions,
};
use envisim_utils::utils::PointSet;

pub use crate::error::EstimationError;
use crate::error::EstimationResult;

/// Returns the balance deviations per dimension
///
/// # Errors
/// Returns an error if a sample unit is oob with respect to the provided data.
#[inline]
fn balance_deviation<I, PROB, DATA>(
    sample: I,
    probabilities: &PROB,
    data: &DATA,
) -> EstimationResult<Vec<f64>>
where
    I: IntoIterator<Item = PROB::Id, IntoIter: Clone>,
    PROB: ProbabilitiesSpec<Real = f64>,
    DATA: PointSet<Id = PROB::Id, Value = f64>,
{
    let sample = sample.into_iter();
    (0..data.dimensions().get())
        .map(|j| {
            // Calculate the dimension total for the population
            let pop_sum = data
                .ids()
                .map(|id| data.coord(id, j))
                .sum::<Option<f64>>()
                .ok_or(EstimationError::InvalidSample)?;
            // Calculate the dimension HT-estimator
            let sample_sum: f64 = sample
                .clone()
                .map(|id| {
                    let p = probabilities
                        .get_real(id)
                        .ok_or(EstimationError::InvalidProbability)?;
                    let x = data.coord(id, j).ok_or(EstimationError::InvalidSample)?;
                    Ok(x / p)
                })
                .sum::<EstimationResult<f64>>()?;
            Ok(pop_sum - sample_sum)
        })
        .collect::<EstimationResult<Vec<f64>>>()
}

/// Calculates the deviation from the spreading matrix.
///
/// # Examples
/// ```
/// # use envisim_estimate::balance::*;
/// # use envisim_utils::matrix::Matrix;
/// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(p)?.set_spreading(m)?;
/// let s = [0, 3, 5, 8, 9];
/// let sb = balance_deviation_spreading(s, &options)?;
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if any sample unit is oob, or the sample is empty.
#[inline]
pub fn balance_deviation_spreading<I, PO, P, BAL>(
    sample: I,
    options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
) -> EstimationResult<Vec<f64>>
where
    I: IntoIterator<Item = PO::Id, IntoIter: Clone>,
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id, Value = f64>,
{
    balance_deviation(sample, options.probabilities(), options.spreading().data())
}

/// Calculates the deviation from the balancing matrix.
///
/// # Examples
/// ```
/// # use envisim_estimate::balance::*;
/// # use envisim_utils::matrix::Matrix;
/// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(p)?.set_balancing(m)?;
/// let s = [0, 3, 5, 8, 9];
/// let sb = balance_deviation_balancing(s, &options)?;
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if any sample unit is oob, or the sample is empty.
#[inline]
pub fn balance_deviation_balancing<I, PO, AUX, P>(
    sample: I,
    options: &SamplingOptions<PO, AUX, BalancingOptions<P>>,
) -> EstimationResult<Vec<f64>>
where
    I: IntoIterator<Item = PO::Id, IntoIter: Clone>,
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id, Value = f64>,
{
    balance_deviation(sample, options.probabilities(), options.balancing().data())
}

#[cfg(test)]
mod tests {
    use envisim_utils::sampling_options::SamplingOptions;
    use envisim_utils::test_utils::*;

    use super::*;

    #[test]
    fn test_balance() {
        let data = Data10::matrix();
        let spec = Data10::prob_e();
        let p = spec.as_real();
        let options = SamplingOptions::with_spec(spec)
            .set_spreading(&data)
            .unwrap();

        let sb = balance_deviation_spreading([0], &options).unwrap();
        let dev = vec![
            data.col_iter(0).unwrap().sum::<f64>() - data[(0, 0)] / p,
            data.col_iter(1).unwrap().sum::<f64>() - data[(0, 1)] / p,
        ];

        assert_vec!(sb, dev);

        let options = options.set_balancing(&data).unwrap();
        let sb = balance_deviation_balancing([0], &options).unwrap();

        assert_vec!(sb, dev);
    }
}
