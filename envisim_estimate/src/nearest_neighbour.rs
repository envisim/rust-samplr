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
use envisim_utils::sampling_options::SpreadingOptions;
use num_traits::ToPrimitive;
use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

use crate::error::{
    EstimationError,
    EstimationResult,
};

/// Nearest neighbour estimator of total.
/// Is not an design-unbiased estimator of the total.
///
/// # Errors
/// Returns an error if the slice lengths dont match, or if the probabilities are invalid.
#[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
#[inline]
pub fn nearest_neighbour<P>(
    y_values: &[f64],
    sample: &[usize],
    auxiliaries: P,
) -> EstimationResult<f64>
where
    P: PointSet<N = f64>,
{
    let population_size = auxiliaries.size().get();
    let sample_size = sample.len();

    if sample.len() != y_values.len() || !sample.iter().all(|id| (0..population_size).contains(id))
    {
        return Err(EstimationError::InvalidSample);
    }

    if sample_size == 0 {
        return Ok(0.0);
    }

    let spr_opts = SpreadingOptions::new(auxiliaries);
    let tree = Tree::new(&spr_opts, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(tree.data());

    let mut number_of_neighbours =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for &id in sample {
        number_of_neighbours.insert(id, 0.0);
    }

    for i in 0..population_size {
        searcher.reset_from_slice(
            &tree
                .data()
                .to_boxed_slice(i)
                .expect("i to exist in aux data"),
        );
        searcher.search(&tree).expect("search to be possible");
        let partial_prob = 1.0
            / searcher
                .neighbours()
                .len()
                .to_f64()
                .expect("neighbour len to convert to f64");

        for n in searcher.neighbours() {
            *number_of_neighbours
                .get_mut(&n.id())
                .expect("neighbour to exsist in hashmap") += partial_prob;
        }
    }

    Ok(y_values
        .iter()
        .zip(sample.iter())
        .fold(0.0, |acc, (&y, id)| acc + y * number_of_neighbours[id]))
}
