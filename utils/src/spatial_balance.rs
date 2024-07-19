use crate::{
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::{Matrix, OperateMatrix, RefMatrix},
    utils::usize_to_f64,
};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

pub fn voronoi<'a>(
    probabilities: &[f64],
    data: &'a RefMatrix,
    sample: &[usize],
    bucket_size: usize,
) -> f64 {
    let population_size = data.nrow();
    let sample_size = sample.len();
    assert!(probabilities.len() == population_size);

    if sample_size == 0 {
        return f64::NAN;
    }

    let mut voronoi_pi =
        FxHashMap::<usize, f64>::with_capacity_and_hasher(sample_size, FxBuildHasher::default());
    assert!(sample.iter().all(|&s| {
        voronoi_pi.insert(s, 0.0);
        s < population_size
    }));

    let tree = Node::new(midpoint_slide, bucket_size, data, &mut sample.to_vec());
    let mut searcher = Searcher::new(&tree, 1);

    for i in 0..population_size {
        searcher.find_neighbours_of_iter(&tree, &mut data.into_row_iter(i));
        let partial_prob = probabilities[i] / usize_to_f64(searcher.neighbours().len());
        searcher.neighbours().iter().for_each(|&s| {
            *voronoi_pi.get_mut(&s).unwrap() += partial_prob;
        });
    }

    let result = voronoi_pi
        .iter()
        .fold(0.0, |acc, (_, &pi)| acc + (pi - 1.0).powi(2));

    result / usize_to_f64(sample_size)
}

pub fn local<'a>(
    probabilities: &[f64],
    data: &'a RefMatrix,
    sample: &[usize],
    bucket_size: usize,
) -> f64 {
    let population_size = data.nrow();
    let sample_size = sample.len();
    assert!(probabilities.len() == population_size);

    if sample_size == 0 {
        return f64::NAN;
    }

    let cols = data.ncol() + 1; // +1 for pi
                                // Store in row-major
    let mut diff_matrix = Matrix::new_fill(0.0, (cols, sample_size));
    let mut norm_matrix = Matrix::new_fill(0.0, (cols, cols * 2));

    for i in 0..cols {
        norm_matrix[(i, i + cols)] = 1.0;
    }

    for (i, &id) in sample.iter().enumerate() {
        assert!(id < population_size, "sample must contain unit indices");
        // Weird p_factor so we can skip tree search later
        let p_factor = (1.0 - probabilities[id]) / probabilities[id];

        diff_matrix[(data.ncol(), i)] = p_factor;
        for j in 0..data.ncol() {
            diff_matrix[(j, i)] = data[(id, j)] * p_factor;
        }
    }

    let mut sample_clone = sample.to_vec();
    let tree = Node::new(midpoint_slide, bucket_size, data, &mut sample_clone);
    let mut searcher = Searcher::new(&tree, 1);
    let sample_set = FxHashSet::<usize>::from_iter(sample_clone);

    for id in 0..population_size {
        norm_matrix[(data.ncol(), data.ncol())] += 1.0;
        for i in 0..data.ncol() {
            for j in 0..data.ncol() {
                norm_matrix[(i, j)] += data[(id, i)] * data[(id, j)];
            }
            norm_matrix[(data.ncol(), i)] += data[(id, i)];
            norm_matrix[(i, data.ncol())] += data[(id, i)];
        }

        if sample_set.contains(&id) {
            continue;
        }

        searcher.find_neighbours_of_iter(&tree, data.into_row_iter(id));

        let share = usize_to_f64(searcher.neighbours().len());
        for &su in searcher.neighbours().iter() {
            diff_matrix[(data.ncol(), su)] -= 1.0 / share;
            for j in 0..data.ncol() {
                diff_matrix[(j, su)] -= data[(id, j)] / share;
            }
        }
    }

    norm_matrix.reduced_row_echelon_form();
    let inv_matrix = RefMatrix::new(
        &norm_matrix.data()[norm_matrix.nrow().pow(2)..],
        norm_matrix.nrow(),
    );

    let mut result = 0.0f64;
    let mut index = 0;
    for _ in 0..sample.len() {
        let vec = &diff_matrix.data()[index..(index + diff_matrix.nrow())];
        result += RefMatrix::new(vec, 1)
            .mult(&inv_matrix)
            .mult(&RefMatrix::new(vec, vec.len()))
            .data()[0];
        index += diff_matrix.nrow();
    }

    (result / usize_to_f64(population_size)).sqrt()
}

#[cfg(test)]
mod tests {
    // use super::*;
    use crate::test_utils::{assert_delta, data_10_2, EPS};

    #[test]
    fn voronoi() {
        let (data, prob) = data_10_2();
        let sb = super::voronoi(&prob, &data, &[0], 3);
        assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2), EPS);
    }

    #[test]
    fn local() {
        let (data, prob) = data_10_2();
        let mut sb: f64;
        sb = super::local(&prob, &data, &[0], 3);
        assert_delta!(sb, 0.9734661634680257247254, EPS);
        sb = super::local(&prob, &data, &[0, 1], 3);
        assert_delta!(sb, 1.251849435249984709984, EPS);
    }
}
