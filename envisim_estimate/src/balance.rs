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

use envisim_samplr::{SampleOptions, SamplingError};
use envisim_utils::InputError;

/// A tuple with the balance deviation from the (spreading, balancing) matrices
pub type BalanceDeviationResult = (Option<Vec<f64>>, Option<Vec<f64>>);

/// Calculates the deviation from the provided auxiliaries.
///
/// Returns a tuple, where the first is the deviation from the spreading auxiliaries, and the second
/// is the deviation from the balancing auxiliaries.
///
/// # Examples
/// ```
/// use envisim_estimate::balance::*;
/// use envisim_samplr::SampleOptions;
/// use envisim_utils::Matrix;
/// use envisim_utils::kd_tree::TreeBuilder;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SampleOptions::new(&p)?.set_spreading(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = balance_deviation(&s, &options)?;
/// # Ok::<(), envisim_samplr::SamplingError>(())
/// ```
pub fn balance_deviation(
    sample: &[usize],
    options: &SampleOptions,
) -> Result<BalanceDeviationResult, SamplingError> {
    options.check_base()?;

    let population_size = options.population_size();

    InputError::check_range_usize(*sample.iter().max().unwrap_or(&0usize), 0, population_size)?;

    let mut bal_spreading: Option<Vec<f64>> = None;
    let mut bal_balancing: Option<Vec<f64>> = None;

    if let Some(spreading) = options.spreading() {
        options.check_spreading()?;
        let data = spreading.data();
        let mut deviation = vec![0.0; data.ncol()];

        for i in 0..options.population_size() {
            for j in 0..data.ncol() {
                deviation[j] += data[(i, j)];
            }
        }

        for &i in sample.iter() {
            let p = options.probabilities()[i];
            for j in 0..data.ncol() {
                deviation[j] -= data[(i, j)] / p;
            }
        }

        bal_spreading = Some(deviation);
    }

    if let Some(balancing) = options.balancing() {
        options.check_balancing()?;
        let data = balancing;
        let mut deviation = vec![0.0; balancing.ncol()];

        for i in 0..options.population_size() {
            for j in 0..data.ncol() {
                deviation[j] += data[(i, j)];
            }
        }

        for &i in sample.iter() {
            let p = options.probabilities()[i];
            for j in 0..data.ncol() {
                deviation[j] -= data[(i, j)] / p;
            }
        }

        bal_balancing = Some(deviation);
    }

    Ok((bal_spreading, bal_balancing))
}
