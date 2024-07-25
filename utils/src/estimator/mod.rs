use crate::error::{InputError, SamplingError};
use crate::kd_tree::{Node, Searcher};
use crate::matrix::{OperateMatrix, RefMatrix};
use crate::utils::usize_to_f64;
use rustc_hash::{FxBuildHasher, FxHashMap};

pub mod hansen_hurwitz;
pub mod horvitz_thompson;

pub fn nearest_neighbour(
    y_values: &[f64],
    sample: &[usize],
    auxilliaries: &RefMatrix,
    bucket_size: usize,
) -> Result<f64, SamplingError> {
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

    let tree = Node::new_midpoint_slide(bucket_size, auxilliaries, &mut sample.to_vec())?;
    let mut searcher = Searcher::new(&tree, 1)?;

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
        .fold(0.0, |acc, (&y, id)| {
            acc + y * number_of_neighbours.get(id).unwrap()
        }))
}
