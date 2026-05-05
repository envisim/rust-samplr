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

use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
};
use envisim_utils::spatial::PointSet;

fn balance_deviation<P>(sample: &[usize], probabilities: &[f64], data: &P) -> Vec<f64>
where
    P: PointSet<f64>,
{
    let population_size = probabilities.len();
    let mut deviation = vec![0.0; data.dim().get()];

    for i in 0..population_size {
        for (j, d) in deviation.iter_mut().enumerate() {
            *d += data.coord(i, j);
        }
    }

    for &i in sample.iter() {
        let p = probabilities[i];
        for (j, d) in deviation.iter_mut().enumerate() {
            *d -= data.coord(i, j) / p;
        }
    }

    deviation
}

/// Calculates the deviation from the spreading matrix.
///
/// # Examples
/// ```
/// # use envisim_estimate::balance::*;
/// # use envisim_utils::sampling_options::*;
/// # use envisim_utils::matrix::Matrix;
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(
///     vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
///     std::num::NonZeroUsize::new(10).unwrap(),
/// ).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = balance_deviation_spreading(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
pub fn balance_deviation_spreading<PS, SOP, M>(
    sample: &[usize],
    options: &SamplingOptions<'_, PS, SOP, M>,
) -> Option<Vec<f64>>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    let population_size = options.population_size().get();

    if !sample.iter().all(|s| (0..population_size).contains(s)) {
        return None;
    }

    let spreading = options.spreading().ok()?;
    Some(balance_deviation(
        sample,
        &options.probabilities().as_f64_slice(),
        spreading.data(),
    ))
}

/// Calculates the deviation from the balancing matrix.
///
/// # Examples
/// ```
/// # use envisim_estimate::balance::*;
/// # use envisim_utils::sampling_options::*;
/// # use envisim_utils::matrix::Matrix;
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(
///     vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
///     std::num::NonZeroUsize::new(10).unwrap(),
/// ).unwrap();
/// let options = SamplingOptions::new(&p)?.set_balancing(m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = balance_deviation_balancing(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
pub fn balance_deviation_balancing<PS, SOP, M>(
    sample: &[usize],
    options: &SamplingOptions<'_, PS, SOP, M>,
) -> Option<Vec<f64>>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    let population_size = options.population_size().get();

    if !sample.iter().all(|s| (0..population_size).contains(s)) {
        return None;
    }

    let balancing = options.spreading().ok()?;
    Some(balance_deviation(
        sample,
        &options.probabilities().as_f64_slice(),
        balancing.data(),
    ))
}
