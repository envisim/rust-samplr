use crate::{
    generate_random::StaticRandom,
    kd_tree::{midpoint_slide, Node, SearcherForNeighbours, Tree, TreeSearch},
    matrix::{OperateMatrix, RefMatrix},
    sampling_controller::*,
    utils::usize_to_f64,
};
use std::collections::HashMap;

const RAND00: StaticRandom = StaticRandom::new(0.0);

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

    let mut tree = Tree::new_with_root(
        DataController::new(&RAND00, probabilities, 0.0, data),
        SearcherForNeighbours::new(data.dim(), 1),
        Box::new(Node::new(
            midpoint_slide,
            bucket_size,
            data,
            &mut sample.to_vec(),
        )),
    );

    for i in 0..population_size {
        tree.find_neighbours_of_unit(&mut data.into_row_iter(i));
        let partial_prob = probabilities[i] / usize_to_f64(tree.searcher().neighbours().len());
        tree.searcher().neighbours().iter().for_each(|&s| {
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
    use crate::{internal_macros::assert_delta, matrix::RefMatrix};

    const DATA_10_2: [f64; 20] = [
        0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
        0.66079779, 0.62911404, 0.06178627, //
        0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
        0.3800352, 0.7774452,
    ];

    fn data_10_2<'a>() -> (RefMatrix<'a>, [f64; 10]) {
        (RefMatrix::new(&DATA_10_2, 10), [0.2f64; 10])
    }

    #[test]
    fn all() {
        let (data, prob) = data_10_2();
        let sb = voronoi(&prob, &data, &[0], 3);
        assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2), 1e-9);
    }
}
