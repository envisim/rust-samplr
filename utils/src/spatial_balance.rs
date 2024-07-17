use crate::{
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::{OperateMatrix, RefMatrix},
    utils::usize_to_f64,
};
use std::collections::HashMap;

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

    let mut voronoi_pi: HashMap<usize, f64> = HashMap::with_capacity(sample_size);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_delta, data_10_2};

    #[test]
    fn all() {
        let (data, prob) = data_10_2();
        let sb = voronoi(&prob, &data, &[0], 3);
        assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2), 1e-9);
    }
}
