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

//! Nearest neighbour estimator

use envisim_utils::kd_tree::searcher::NearestNeighbourSearcher;
use envisim_utils::kd_tree::{
    PointSet,
    Tree,
};
use envisim_utils::sampling_options::{
    SamplingOptionsError,
    SpreadingOptions,
};
use envisim_utils::utils::usize_to_f64;
use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

/// Nearest neighbour estimator of total.
/// Is not an design-unbiased estimator of the total.
pub fn nearest_neighbour<P>(
    y_values: &[f64],
    sample: &[usize],
    auxiliaries: P,
) -> Result<f64, SamplingOptionsError>
where
    P: PointSet<f64>,
{
    let population_size = auxiliaries.size().get();
    let sample_size = sample.len();

    if sample.len() != y_values.len() || !sample.iter().all(|id| (0..population_size).contains(id))
    {
        return Err(SamplingOptionsError::InvalidSample);
    }

    if sample_size == 0 {
        return Ok(0.0);
    }

    let spr_opts = SpreadingOptions::new(auxiliaries)?;
    let tree = Tree::new(&spr_opts, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(tree.data());

    let mut number_of_neighbours =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for &id in sample.iter() {
        number_of_neighbours.insert(id, 0.0);
    }

    for i in 0..population_size {
        searcher.reset_from_slice(&tree.data().to_boxed_slice(i).unwrap());
        searcher.search(&tree).unwrap();
        let partial_prob = 1.0 / usize_to_f64(searcher.neighbours().len());

        for n in searcher.neighbours().iter() {
            *number_of_neighbours.get_mut(&n.id()).unwrap() += partial_prob;
        }
    }

    Ok(y_values
        .iter()
        .zip(sample.iter())
        .fold(0.0, |acc, (&y, id)| acc + y * number_of_neighbours[id]))
}
