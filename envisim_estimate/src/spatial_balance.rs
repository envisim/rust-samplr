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

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::NearestNeighbourSearcher;
use envisim_utils::matrix::{
    Matrix,
    MatrixDims,
    MatrixRef,
    PointSet,
};
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
    SamplingOptionsError,
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
/// # use envisim_estimate::spatial_balance::*;
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
/// // let sb = voronoi(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Grafström, A., & Schelin, L. (2014).
/// How to select representative samples.
/// Scandinavian Journal of Statistics, 41(2), 277-290.
/// <https://doi.org/10.1111/sjos.12016>
pub fn voronoi<PS, SOP, M>(
    sample: &[usize],
    options: &SamplingOptions<PS, SOP, M>,
) -> Result<f64, SamplingOptionsError>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    let sample_size = sample.len();
    if sample_size == 0 {
        return Ok(f64::NAN);
    }

    let tree = Tree::new(options.spreading()?, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(&tree.data());

    let mut voronoi_pi =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample.len(), FxBuildHasher);
    for &id in sample.iter() {
        if voronoi_pi.insert(id, 0.0).is_some() {
            return Err(SamplingOptionsError::InvalidSample);
        }
    }

    if let Some(spec) = options.probabilities().as_equal() {
        let population_size = spec.population_size().get();
        let p = spec.as_f64();
        for i in 0..population_size {
            searcher.reset_from_slice(&tree.data().to_boxed_slice(i).unwrap());
            searcher.search(&tree).unwrap();
            let partial_prob = p / usize_to_f64(searcher.neighbours().len());
            for n in searcher.neighbours().iter() {
                *voronoi_pi.get_mut(&n.id()).unwrap() += partial_prob;
            }
        }
    } else {
        let values = options.probabilities().as_f64_slice();
        for (i, &p) in values.iter().enumerate() {
            searcher.reset_from_slice(&tree.data().to_boxed_slice(i).unwrap());
            searcher.search(&tree).unwrap();
            let partial_prob = p / usize_to_f64(searcher.neighbours().len());
            for n in searcher.neighbours().iter() {
                *voronoi_pi.get_mut(&n.id()).unwrap() += partial_prob;
            }
        }
    }

    let result = voronoi_pi
        .iter()
        .fold(0.0, |acc, (_, &pi)| acc + (pi - 1.0).powi(2));

    Ok(result / usize_to_f64(sample.len()))
}

/// Local measure of spatial balance.
///
/// # Examples
/// ```
/// # use envisim_estimate::spatial_balance::*;
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
/// let sb = local(&s, &options, true).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Prentius, W., & Grafström, A. (2024).
/// How to find the best sampling design: A new measure of spatial balance.
/// Environmetrics, e2878.
/// <https://doi.org/10.1002/env.2878>
pub fn local<PS, SOP, M>(
    sample: &[usize],
    options: &SamplingOptions<PS, SOP, M>,
    balance_probabilities: bool,
) -> Result<f64, SamplingOptionsError>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    if sample.is_empty() {
        return Ok(f64::NAN);
    }

    let tree = Tree::new(options.spreading()?, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(&tree.data());

    let population_size = options.population_size().get();

    // One extra column for inclusion probabilities
    let cols = tree.data().dim().get() + if balance_probabilities { 1 } else { 0 };
    let mut voronoi_means =
        FxHashMap::<usize, Vec<f64>>::with_capacity_and_hasher(sample.len(), FxBuildHasher);

    // The gram matrix
    let mut norm_matrix = Matrix::from_value(0.0, MatrixDims::try_new(cols, cols * 2).unwrap());

    for i in 0..cols {
        norm_matrix[(i, i + cols)] = 1.0;
    }

    if let Some(spec) = options.probabilities().as_equal() {
        let p = spec.as_f64();
        let p_factor = (1.0 - p) / p;
        for &id in sample.iter() {
            // Weird p_factor so we can skip tree search later
            let mut mean = vec![p_factor; cols];

            for (j, m) in mean.iter_mut().enumerate().take(tree.data().dim().get()) {
                *m *= tree.data().coord(id, j);
            }

            if voronoi_means.insert(id, mean).is_some() {
                return Err(SamplingOptionsError::InvalidSample);
            }
        }
    } else {
        let values = options.probabilities().as_f64_slice();
        for &id in sample.iter() {
            // Weird p_factor so we can skip tree search later
            let p_factor = (1.0 - values[id]) / values[id];
            let mut mean = vec![p_factor; cols];

            for (j, m) in mean.iter_mut().enumerate().take(tree.data().dim().get()) {
                *m *= tree.data().coord(id, j);
            }

            if voronoi_means.insert(id, mean).is_some() {
                return Err(SamplingOptionsError::InvalidSample);
            }
        }
    }

    for id in 0..population_size {
        if balance_probabilities {
            norm_matrix[(tree.data().dim().get(), tree.data().dim().get())] += 1.0;
        }

        for i in 0..tree.data().dim().get() {
            for j in 0..tree.data().dim().get() {
                norm_matrix[(i, j)] += tree.data().coord(id, i) * tree.data().coord(id, j);
            }

            if balance_probabilities {
                norm_matrix[(tree.data().dim().get(), i)] += tree.data().coord(id, i);
                norm_matrix[(i, tree.data().dim().get())] += tree.data().coord(id, i);
            }
        }

        // We have already added the sample units, so we can skip this
        // Has an edge case, where two sample units are exactly overlapping. In this case, this
        // implementation assumes that the "self" sample unit is the sole voronoi cluster, rather
        // than sharing.
        if voronoi_means.contains_key(&id) {
            continue;
        }

        searcher.reset_from_slice(&tree.data().to_boxed_slice(id).unwrap());
        searcher.search(&tree);

        let share = usize_to_f64(searcher.neighbours().len());
        for &n in searcher.neighbours().iter() {
            let mean = voronoi_means.get_mut(&n.id()).unwrap();

            for (j, m) in mean.iter_mut().enumerate().take(tree.data().dim().get()) {
                *m -= tree.data().coord(id, j) / share;
            }

            if balance_probabilities {
                mean[tree.data().dim().get()] -= 1.0 / share;
            }
        }
    }

    norm_matrix.reduced_row_echelon_form();
    let inv_matrix = Matrix::new(
        norm_matrix.data()[norm_matrix.nrow().get().pow(2)..].to_vec(),
        norm_matrix.nrow(),
    )
    .unwrap();

    let result = voronoi_means.iter().fold(0.0, |acc, (_, vec)| {
        acc + MatrixRef::new(vec, NonZeroUsize::new(1).unwrap())
            .unwrap()
            .mul_mat(&inv_matrix)
            .unwrap()
            .mul_mat(&MatrixRef::new(vec, NonZeroUsize::new(cols).unwrap()).unwrap())
            .unwrap()
            .data()[0]
    });

    Ok((result / usize_to_f64(population_size)).sqrt())
}

/// Energy distance between sample distribution and population.
///
/// # Examples
/// ```
/// # use envisim_estimate::spatial_balance::*;
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
/// let sb = energy_distance(&s, &options).unwrap();
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
pub fn energy_distance<PS, SOP, M>(
    sample: &[usize],
    options: &SamplingOptions<PS, SOP, M>,
) -> Result<f64, SamplingOptionsError>
where
    PS: ProbabilitySpec,
    SOP: PointSet<f64>,
{
    let matrix = options.spreading()?.data();

    let (phi, u_spread) = if options.probabilities().as_equal().is_some() {
        energy_distance_phi_equal(matrix)
    } else {
        let values = options.probabilities().as_f64_slice();
        let s_size = usize_to_f64(sample.len());
        energy_distance_phi_unequal(matrix, &values, s_size)
    };

    let edi = energy_distance_internal(sample, matrix, &phi);
    let distance = edi - u_spread;

    Ok(distance)
}

/// Returns (phi-vec, phi-sumish)
pub(crate) fn energy_distance_phi_equal<P>(matrix: &P) -> (Vec<f64>, f64)
where
    P: PointSet<f64>,
{
    let size = matrix.size().get();
    let u_size = usize_to_f64(size);
    let mut phi = vec![0.0; size];
    let mut u_spread = 0.0;

    for id1 in matrix.id_iter() {
        for id2 in matrix.id_iter().skip(id1 + 1) {
            let dist = matrix.sq_distance_between(id1, id2).sqrt();
            phi[id1] += dist;
            phi[id2] += dist;
        }
        phi[id1] /= u_size;
        u_spread += phi[id1];
    }
    (phi, u_spread / u_size)
}
/// Returns (phi-vec, phi-sumish)
pub(crate) fn energy_distance_phi_unequal<P>(
    matrix: P,
    probabilities: &[f64],
    s_size: f64,
) -> (Vec<f64>, f64)
where
    P: PointSet<f64>,
{
    let size = matrix.size().get();
    let mut phi = vec![0.0; size];
    let mut u_spread = 0.0;

    for id1 in matrix.id_iter() {
        for id2 in matrix.id_iter().skip(id1 + 1) {
            let dist = matrix.sq_distance_between(id1, id2).sqrt();
            phi[id1] += dist * probabilities[id2] / s_size;
            phi[id2] += dist * probabilities[id1] / s_size;
        }
        u_spread += phi[id1] * probabilities[id1];
    }
    (phi, u_spread / s_size)
}

/// Returns 2 E||X-Z|| - E||X-X'||
pub(crate) fn energy_distance_internal<P>(sample: &[usize], matrix: P, phi: &[f64]) -> f64
where
    P: PointSet<f64>,
{
    let s_size = usize_to_f64(sample.len());
    let mut s_spread: f64 = 0.0;
    let mut inter_spread: f64 = 0.0;

    for i in 0..sample.len() {
        let id1 = sample[i];
        inter_spread += phi[id1];

        // Iterate over 0..i
        for &id2 in sample.iter().take(i) {
            s_spread += 2.0 * matrix.sq_distance_between(id1, id2).sqrt();
        }
    }

    inter_spread /= s_size;
    s_spread /= s_size.powi(2);
    inter_spread * 2.0 - s_spread
}

#[cfg(test)]
mod test {
    use envisim_test_utils::*;

    use super::*;

    #[test]
    fn ed_phi() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = MatrixRef::new(&m_data, NonZeroUsize::new(3).unwrap()).unwrap();
        let phi = energy_distance_phi_equal(&data);
        let res: Vec<f64> = vec![
            (2.0f64.sqrt() + 8.0f64.sqrt()) / 3.0f64,
            (2.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
            (8.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
        ];

        assert_fvec(&phi.0, &res);
    }
    #[test]
    fn ed_internal() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = MatrixRef::new(&m_data, NonZeroUsize::new(3).unwrap()).unwrap();
        let phi = energy_distance_phi_equal(&data);
        let dist = energy_distance_internal(&[1, 2], &data, &phi.0);
        let res: f64 = 2.0 * (phi.0[1] + phi.0[2]) / 2.0 - (2.0f64.sqrt() + 2.0f64.sqrt()) / 4.0;

        assert_delta!(dist, res);
    }
}
