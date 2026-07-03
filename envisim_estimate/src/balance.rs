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
    ProbabilityOptions,
    SpreadingOptions,
};
use envisim_utils::spatial::PointSet;

pub use crate::error::EstimationError;
use crate::error::EstimationResult;
use crate::utils::ypi_quotient;

/// Returns the balance deviations per dimension
///
/// # Errors
/// Returns an error if a sample unit is oob with respect to the provided data.
#[expect(
    clippy::needless_pass_by_value,
    reason = "iterator pass by value is ok"
)]
#[inline]
fn balance_deviation<I, P>(
    sample: &[usize],
    probabilities: I,
    data: &P,
) -> EstimationResult<Vec<f64>>
where
    I: ExactSizeIterator<Item = f64> + Clone,
    P: PointSet<Id = usize, Value = f64>,
{
    let population_size = probabilities.len();

    (0..data.dimensions().get())
        .map(|j| {
            // Calculate the dimension total for the population
            let pop_sum: f64 = (0..population_size)
                .map(|i| data.get_coord(i, j))
                .sum::<Option<f64>>()
                .ok_or(EstimationError::InvalidSample)?;
            // Calculate the dimension HT-estimator
            let sample_sum: f64 = sample
                .iter()
                .map(|&i| {
                    let p = probabilities.clone().nth(i);
                    data.get_coord(i, j)
                        .zip(p)
                        .ok_or(EstimationError::InvalidSample)
                        .and_then(ypi_quotient)
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
/// let sb = balance_deviation_spreading(&s, &options)?;
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if any sample unit is oob, or the sample is empty.
#[inline]
pub fn balance_deviation_spreading<PO, P, BAL>(
    sample: &[usize],
    options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
) -> EstimationResult<Vec<f64>>
where
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize, Value = f64>,
{
    balance_deviation(
        sample,
        options.probabilities().iter_real(),
        options.spreading().data(),
    )
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
/// let sb = balance_deviation_balancing(&s, &options)?;
/// # Ok::<(), EstimationError>(())
/// ```
///
/// # Errors
/// Returns an error if any sample unit is oob, or the sample is empty.
#[inline]
pub fn balance_deviation_balancing<PO, AUX, P>(
    sample: &[usize],
    options: &SamplingOptions<PO, AUX, BalancingOptions<P>>,
) -> EstimationResult<Vec<f64>>
where
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize, Value = f64>,
{
    balance_deviation(
        sample,
        options.probabilities().iter_real(),
        options.balancing().data(),
    )
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
        let options = SamplingOptions::with_spec_equal(spec)
            .set_spreading(&data)
            .unwrap();

        let sb = balance_deviation_spreading(&[0], &options).unwrap();
        let dev = vec![
            data.col_iter(0).unwrap().sum::<f64>() - data[(0, 0)] / p,
            data.col_iter(1).unwrap().sum::<f64>() - data[(0, 1)] / p,
        ];

        assert_vec!(sb, dev);

        let options = options.set_balancing(&data).unwrap();
        let sb = balance_deviation_balancing(&[0], &options).unwrap();

        assert_vec!(sb, dev);
    }
}
