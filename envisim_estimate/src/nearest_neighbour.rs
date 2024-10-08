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

//! Nearest neighbour estimator

use envisim_samplr::SamplingError;
use envisim_utils::kd_tree::{Searcher, TreeBuilder};
use envisim_utils::utils::usize_to_f64;
use envisim_utils::InputError;
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Nearest neighbour estimator of total.
/// Is not an design-unbiased estimator of the total.
pub fn nearest_neighbour(
    y_values: &[f64],
    sample: &[usize],
    tree_builder: &TreeBuilder,
) -> Result<f64, SamplingError> {
    let tree = tree_builder.build(&mut sample.to_vec())?;
    let mut searcher = Searcher::new_1(&tree);
    let auxilliaries = tree.data();

    let population_size = auxilliaries.nrow();
    let sample_size = sample.len();
    InputError::check_lengths(y_values, sample).and(InputError::check_sizes(
        population_size,
        auxilliaries.nrow(),
    ))?;

    let mut number_of_neighbours =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for &id in sample.iter() {
        number_of_neighbours.insert(id, 0.0);
    }

    for i in 0..population_size {
        searcher
            .find_neighbours_of_iter(&tree, &mut auxilliaries.row_iter(i))
            .unwrap();
        let part = 1.0 / usize_to_f64(searcher.neighbours().len());

        for id in searcher.neighbours().iter() {
            *number_of_neighbours.get_mut(id).unwrap() += part;
        }
    }

    Ok(y_values
        .iter()
        .zip(sample.iter())
        .fold(0.0, |acc, (&y, id)| acc + y * number_of_neighbours[id]))
}
