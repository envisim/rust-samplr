use std::num::NonZeroUsize;

use envisim_test_utils::*;
use envisim_utils::kd_tree::*;
use envisim_utils::matrix::Matrix;
use envisim_utils::probabilities::*;
use envisim_utils::sampling_options::{
    SamplingOptionsError,
    SpreadingOptions,
};

const MATRIX_DATA: [f64; 10] = [
    0.0, 1.0, 2.0, 13.0, 14.0, //
    0.0, 10.0, 20.0, 30.0, 40.0, //
];

fn matrix_new<'a>() -> Matrix<'a> { Matrix::new(&MATRIX_DATA, 5).unwrap() }

#[test]
fn searcher() -> Result<(), SamplingOptionsError> {
    let m = matrix_new();
    let opts = SpreadingOptions::new(m).set_bucket_size(2)?;
    let t = opts.build(&mut [0, 1, 2, 3]).unwrap();

    let mut s = Searcher::new_1(&t);
    s.find_neighbours(&t, &vec![5.0, 5.0]).unwrap();
    assert_eq!(s.neighbours(), vec![1]);
    assert_delta!(s.distance_k(0), 41.0);

    let mut s = Searcher::new(&t, NonZeroUsize::new(2).unwrap());
    s.find_neighbours_of_id(&t, 3).unwrap();
    assert_eq!(s.neighbours(), vec![2, 1]);
    assert_delta!(s.distance_k(0), 221.0);
    assert_delta!(s.distance_k(1), 544.0);

    Ok(())
}

#[test]
fn searcher_weighted() -> Result<(), SamplingOptionsError> {
    let m = matrix_new();
    let opts = SpreadingOptions::new(m).set_bucket_size(2)?;
    let t = opts.build(&mut [0, 1, 2, 3, 4]).unwrap();
    let p = ProbabilitiesUnequal::with_value(5, 0.25, 1e-12).unwrap();

    let mut s = SearcherWeighted::new(&t);
    s.find_neighbours(&t, &p, &vec![5.0, 5.0], 0.5).unwrap();
    assert_eq!(s.neighbours(), vec![1, 0]);
    assert_delta!(s.weight_k(0), 0.5);
    assert_delta!(s.weight_k(1), 0.5);

    s.find_neighbours_of_id(&t, &p, 3).unwrap();
    assert_eq!(s.neighbours(), vec![4, 2, 1]);
    assert_delta!(s.weight_k(0), 1.0 / 3.0);
    assert_delta!(s.weight_k(1), 1.0 / 3.0);
    assert_delta!(s.weight_k(2), 1.0 / 3.0);

    Ok(())
}
