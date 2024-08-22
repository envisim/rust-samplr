// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

pub use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::kd_tree::{Searcher, TreeBuilder};
use envisim_utils::matrix::{Matrix, OperateMatrix, RefMatrix};
use envisim_utils::utils::usize_to_f64;
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Voronoi measure of spatial balance.
///
/// # Examples
/// ```
/// use envisim_estimate::spatial_balance::*;
/// use envisim_utils::matrix::RefMatrix;
/// use envisim_utils::kd_tree::TreeBuilder;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let dt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// let m = RefMatrix::new(&dt, 10);
/// let s = [0, 3, 5, 8, 9];
///
/// // let sb = voronoi(&s, &p, &TreeBuilder::new(&m))?;
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., & Schelin, L. (2014).
/// How to select representative samples.
/// Scandinavian Journal of Statistics, 41(2), 277-290.
/// <https://doi.org/10.1111/sjos.12016>
pub fn voronoi(
    sample: &[usize],
    probabilities: &[f64],
    tree_builder: &TreeBuilder,
) -> Result<f64, SamplingError> {
    let tree = tree_builder.build(&mut sample.to_vec())?;
    let mut searcher = Searcher::new(&tree, 1)?;
    let data = tree.data();

    let population_size = data.nrow();
    let sample_size = sample.len();
    InputError::check_sizes(probabilities.len(), population_size)?;

    if sample_size == 0 {
        return Ok(f64::NAN);
    }

    let mut voronoi_pi =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher);
    for &id in sample.iter() {
        voronoi_pi.insert(id, 0.0);
    }

    for (i, &p) in probabilities.iter().enumerate() {
        searcher
            .find_neighbours_of_iter(&tree, &mut data.row_iter(i))
            .unwrap();
        let partial_prob = p / usize_to_f64(searcher.neighbours().len());
        searcher.neighbours().iter().for_each(|&s| {
            *voronoi_pi.get_mut(&s).unwrap() += partial_prob;
        });
    }

    let result = voronoi_pi
        .iter()
        .fold(0.0, |acc, (_, &pi)| acc + (pi - 1.0).powi(2));

    Ok(result / usize_to_f64(sample_size))
}

/// Local measure of spatial balance.
///
/// # Examples
/// ```
/// use envisim_estimate::spatial_balance::*;
/// use envisim_utils::matrix::RefMatrix;
/// use envisim_utils::kd_tree::TreeBuilder;
///
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let dt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// let m = RefMatrix::new(&dt, 10);
/// let s = [0, 3, 5, 8, 9];
///
/// let sb = local(&s, &p, &TreeBuilder::new(&m))?;
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Prentius, W., & Grafström, A. (2024).
/// How to find the best sampling design: A new measure of spatial balance.
/// Environmetrics, e2878.
/// <https://doi.org/10.1002/env.2878>
pub fn local(
    sample: &[usize],
    probabilities: &[f64],
    tree_builder: &TreeBuilder,
) -> Result<f64, SamplingError> {
    let tree = tree_builder.build(&mut sample.to_vec())?;
    let mut searcher = Searcher::new(&tree, 1)?;
    let data = tree.data();

    let population_size = data.nrow();
    let sample_size = sample.len();
    InputError::check_sizes(probabilities.len(), population_size)?;

    if sample_size == 0 {
        return Ok(f64::NAN);
    }

    // One extra column for inclusion probabilitoies
    let cols = data.ncol() + 1;
    let mut voronoi_means =
        FxHashMap::<usize, Vec<f64>>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    // The gram matrix
    let mut norm_matrix = Matrix::new_fill(0.0, (cols, cols * 2));

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
            return Err(SamplingError::Input(InputError::NotUnique));
        }
    }

    for id in 0..population_size {
        norm_matrix[(data.ncol(), data.ncol())] += 1.0;
        for i in 0..data.ncol() {
            for j in 0..data.ncol() {
                norm_matrix[(i, j)] += data[(id, i)] * data[(id, j)];
            }
            norm_matrix[(data.ncol(), i)] += data[(id, i)];
            norm_matrix[(i, data.ncol())] += data[(id, i)];
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
            mean[data.ncol()] -= 1.0 / share;
        }
    }

    norm_matrix.reduced_row_echelon_form();
    let inv_matrix = RefMatrix::new(
        &norm_matrix.data()[norm_matrix.nrow().pow(2)..],
        norm_matrix.nrow(),
    );

    let result = voronoi_means.iter().fold(0.0, |acc, (_, vec)| {
        acc + RefMatrix::new(vec, 1)
            .mult(&inv_matrix)
            .mult(&RefMatrix::new(vec, cols))
            .data()[0]
    });

    Ok((result / usize_to_f64(population_size)).sqrt())
}
