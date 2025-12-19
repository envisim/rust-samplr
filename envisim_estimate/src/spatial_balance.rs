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

//! Spatial balance measures

use envisim_utils::kd_tree::{
    Searcher,
    TreeBuilder,
};
use envisim_utils::matrix::Matrix;
use envisim_utils::probabilities::Probabilities;
use envisim_utils::sampling_options::{
    Enabled,
    ProbabilitySpec,
    SamplingOptions,
};
use envisim_utils::utils::usize_to_f64;
use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

/// Voronoi measure of spatial balance.
///
/// # Examples
/// ```
/// use envisim_estimate::spatial_balance::*;
/// use envisim_utils::sampling_options::*;
/// use envisim_utils::matrix::Matrix;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = voronoi(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Grafström, A., & Schelin, L. (2014).
/// How to select representative samples.
/// Scandinavian Journal of Statistics, 41(2), 277-290.
/// <https://doi.org/10.1111/sjos.12016>
pub fn voronoi<P, B>(sample: &[usize], options: &SamplingOptions<P, Enabled, B>) -> Option<f64>
where
    P: Probabilities,
{
    let tree = options.spreading().build(&mut sample.to_vec()).unwrap();
    let mut searcher = Searcher::new_1(&tree);
    let data = tree.data();
    let probabilities = options.probabilities().slice();

    let sample_size = sample.len();

    if sample_size == 0 {
        return Some(f64::NAN);
    }

    let mut voronoi_pi =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher);
    for &id in sample.iter() {
        if voronoi_pi.insert(id, 0.0).is_some() {
            return None;
        }
    }

    for (i, &p) in probabilities.iter().enumerate() {
        searcher
            .find_neighbours_of_iter(&tree, data.row_iter(i))
            .unwrap();
        let partial_prob = p / usize_to_f64(searcher.neighbours().len());
        searcher.neighbours().iter().for_each(|&s| {
            *voronoi_pi.get_mut(&s).unwrap() += partial_prob;
        });
    }

    let result = voronoi_pi
        .iter()
        .fold(0.0, |acc, (_, &pi)| acc + (pi - 1.0).powi(2));

    Some(result / usize_to_f64(sample_size))
}

/// Local measure of spatial balance.
///
/// # Examples
/// ```
/// use envisim_estimate::spatial_balance::*;
/// use envisim_utils::sampling_options::*;
/// use envisim_utils::matrix::Matrix;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// let sb = local(&s, &options, true).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Prentius, W., & Grafström, A. (2024).
/// How to find the best sampling design: A new measure of spatial balance.
/// Environmetrics, e2878.
/// <https://doi.org/10.1002/env.2878>
pub fn local<P, B>(
    sample: &[usize],
    options: &SamplingOptions<P, Enabled, B>,
    balance_probabilities: bool,
) -> Option<f64>
where
    P: Probabilities,
{
    let tree = options.spreading().build(&mut sample.to_vec()).unwrap();
    let mut searcher = Searcher::new_1(&tree);
    let data = tree.data();
    let probabilities = options.probabilities().slice();

    let population_size = options.population_size();
    let sample_size = sample.len();

    if sample_size == 0 {
        return Some(f64::NAN);
    }

    // One extra column for inclusion probabilities
    let cols = data.ncol() + if balance_probabilities { 1 } else { 0 };
    let mut voronoi_means =
        FxHashMap::<usize, Vec<f64>>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    // The gram matrix
    let mut norm_matrix = Matrix::from_value(0.0, (cols, cols * 2)).unwrap();

    for i in 0..cols {
        norm_matrix[(i, i + cols)] = 1.0;
    }

    for &id in sample.iter() {
        // Weird p_factor so we can skip tree search later
        let p_factor = (1.0 - probabilities[id]) / probabilities[id];
        let mut mean = vec![p_factor; cols];

        for (i, v) in data.row_iter(id).enumerate() {
            mean[i] *= v;
        }

        if voronoi_means.insert(id, mean).is_some() {
            return None;
        }
    }

    for id in 0..population_size {
        if balance_probabilities {
            norm_matrix[(data.ncol(), data.ncol())] += 1.0;
        }

        for i in 0..data.ncol() {
            for j in 0..data.ncol() {
                norm_matrix[(i, j)] += data[(id, i)] * data[(id, j)];
            }

            if balance_probabilities {
                norm_matrix[(data.ncol(), i)] += data[(id, i)];
                norm_matrix[(i, data.ncol())] += data[(id, i)];
            }
        }

        // We have already added the sample units, so we can skip this
        // Has an edge case, where two sample units are exactly overlapping. In this case, this
        // implementation assumes that the "self" sample unit is the sole voronoi cluster, rather
        // than sharing.
        if voronoi_means.contains_key(&id) {
            continue;
        }

        searcher
            .find_neighbours_of_iter(&tree, data.row_iter(id))
            .unwrap();

        let share = usize_to_f64(searcher.neighbours().len());
        for &su in searcher.neighbours().iter() {
            let mean = voronoi_means.get_mut(&su).unwrap();
            for (i, v) in data.row_iter(id).enumerate() {
                mean[i] -= v / share;
            }

            if balance_probabilities {
                mean[data.ncol()] -= 1.0 / share;
            }
        }
    }

    norm_matrix.reduced_row_echelon_form();
    let inv_matrix = Matrix::new(
        &norm_matrix.data()[norm_matrix.nrow().pow(2)..],
        norm_matrix.nrow(),
    )
    .unwrap();

    let result = voronoi_means.iter().fold(0.0, |acc, (_, vec)| {
        acc + Matrix::new(vec, 1)
            .unwrap()
            .mult(&inv_matrix)
            .unwrap()
            .mult(&Matrix::new(vec, cols).unwrap())
            .unwrap()
            .data()[0]
    });

    Some((result / usize_to_f64(population_size)).sqrt())
}

/// Energy distance between sample distribution and population.
///
/// # Examples
/// ```
/// use envisim_estimate::spatial_balance::*;
/// use envisim_utils::sampling_options::*;
/// use envisim_utils::matrix::Matrix;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(&m)?;
/// let s = [0, 3, 5, 8, 9];
///
/// let sb = energy_distance(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
pub fn energy_distance<P, B>(
    sample: &[usize],
    options: &SamplingOptions<P, Enabled, B>,
) -> Option<f64>
where
    P: Probabilities,
{
    let org_matrix = options.spreading().data();

    let mut matrix = org_matrix.clone();
    matrix.to_mut();

    let distance: f64 = match options.probabilities().spec() {
        ProbabilitySpec::Unequal { ref values } => {
            for unit in 0usize..matrix.nrow() {
                let prob = values[unit];

                for j in 0usize..matrix.ncol() {
                    matrix[(unit, j)] /= prob;
                }
            }

            let phi = energy_distance_phi(&matrix);
            energy_distance_internal(sample, &matrix, &phi)
        }
        _ => {
            let u_size = usize_to_f64(matrix.nrow());
            let s_size = usize_to_f64(sample.len());
            let phi = energy_distance_phi(&matrix);
            energy_distance_internal(sample, &matrix, &phi) * u_size / s_size
        }
    };

    Some(distance)
}

pub(crate) fn energy_distance_phi(matrix: &Matrix) -> Vec<f64> {
    let u_size = usize_to_f64(matrix.nrow());
    let mut phi = vec![0.0; matrix.nrow()];

    for id1 in 0..matrix.nrow() {
        for id2 in (id1 + 1)..matrix.nrow() {
            let dist = matrix.distance_between_rows(id1, id2).unwrap().sqrt();
            phi[id1] += dist;
            phi[id2] += dist;
        }
        phi[id1] /= u_size;
    }
    phi
}

pub(crate) fn energy_distance_internal(sample: &[usize], matrix: &Matrix<'_>, phi: &[f64]) -> f64 {
    let u_size = usize_to_f64(matrix.nrow());
    let s_size = usize_to_f64(sample.len());
    let u_spread: f64 = phi.iter().sum::<f64>() / u_size;
    let mut s_spread: f64 = 0.0;
    let mut inter_spread: f64 = 0.0;

    for i in 0..sample.len() {
        let id1 = sample[i];
        inter_spread += phi[id1];

        // Iterate over 0..i
        for &id2 in sample.iter().take(i) {
            s_spread += 2.0 * matrix.distance_between_rows(id1, id2).unwrap().sqrt();
        }
    }

    inter_spread /= s_size;
    s_spread /= s_size.powi(2);
    inter_spread * 2.0 - u_spread - s_spread
}

#[cfg(test)]
mod test {
    use envisim_test_utils::*;

    use super::*;

    #[test]
    fn ed_phi() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Matrix::new(&m_data, 3).unwrap();
        let phi = energy_distance_phi(&data);
        let res: Vec<f64> = vec![
            (2.0f64.sqrt() + 8.0f64.sqrt()) / 3.0f64,
            (2.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
            (8.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
        ];

        assert_fvec(&phi, &res);
    }
    #[test]
    fn ed_internal() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Matrix::new(&m_data, 3).unwrap();
        let phi = energy_distance_phi(&data);
        let dist = energy_distance_internal(&[1, 2], &data, &phi);
        let res: f64 = 2.0 * (phi[0] + phi[1]) / 2.0
            - (2.0f64.sqrt() + 2.0f64.sqrt()) / 4.0
            - phi.iter().sum::<f64>() / 3.0; //

        assert_delta!(dist, res);
    }
}
