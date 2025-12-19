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

//! Balance deviation

use envisim_utils::matrix::Matrix;
use envisim_utils::probabilities::Probabilities;
use envisim_utils::sampling_options::{
    Enabled,
    SamplingOptions,
};

/// A tuple with the balance deviation from the (spreading, balancing) matrices
pub type BalanceDeviationResult = (Option<Vec<f64>>, Option<Vec<f64>>);

fn balance_deviation(sample: &[usize], probabilities: &[f64], data: &Matrix) -> Vec<f64> {
    let population_size = probabilities.len();
    let mut deviation = vec![0.0; data.ncol()];

    for i in 0..population_size {
        for j in 0..data.ncol() {
            deviation[j] += data[(i, j)];
        }
    }

    for &i in sample.iter() {
        let p = probabilities[i];
        for j in 0..data.ncol() {
            deviation[j] -= data[(i, j)] / p;
        }
    }

    deviation
}

/// Calculates the deviation from the spreading matrix.
///
/// # Examples
/// ```
/// use envisim_estimate::balance::*;
/// use envisim_utils::sampling_options::*;
/// use envisim_utils::matrix::Matrix;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = balance_deviation_spreading(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
pub fn balance_deviation_spreading<P, B>(
    sample: &[usize],
    options: &SamplingOptions<'_, P, Enabled, B>,
) -> Option<Vec<f64>>
where
    P: Probabilities,
{
    let population_size = options.population_size();

    if !sample.iter().all(|s| (0..population_size).contains(s)) {
        return None;
    }

    Some(balance_deviation(
        sample,
        &options.probabilities().slice(),
        options.spreading().data(),
    ))
}

/// Calculates the deviation from the balancing matrix.
///
/// # Examples
/// ```
/// use envisim_estimate::balance::*;
/// use envisim_utils::sampling_options::*;
/// use envisim_utils::matrix::Matrix;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_balancing(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = balance_deviation_balancing(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
pub fn balance_deviation_balancing<P, S>(
    sample: &[usize],
    options: &SamplingOptions<'_, P, S, Enabled>,
) -> Option<Vec<f64>>
where
    P: Probabilities,
{
    let population_size = options.population_size();

    if !sample.iter().all(|s| (0..population_size).contains(s)) {
        return None;
    }

    Some(balance_deviation(
        sample,
        &options.probabilities().slice(),
        options.balancing().data(),
    ))
}
