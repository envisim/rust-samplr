use envisim_test_utils::*;
use envisim_utils::kd_tree::*;
use envisim_utils::{Matrix, Probabilities};

fn matrix_new<'a>() -> Matrix<'a> {
    Matrix::new(
        &[
            0.0, 1.0, 2.0, 13.0, 14.0, //
            0.0, 10.0, 20.0, 30.0, 40.0, //
        ],
        5,
    )
}

#[test]
fn searcher() -> Result<(), NodeError> {
    let m = matrix_new();
    let t = TreeBuilder::new(&m)
        .try_bucket_size(2)?
        .build(&mut [0, 1, 2, 3])?;

    let mut s = Searcher::new(&t, 1).unwrap();
    s.find_neighbours(&t, &vec![5.0, 5.0]).unwrap();
    assert_eq!(s.neighbours(), vec![1]);
    assert_delta!(s.distance_k(0), 41.0);

    let mut s = Searcher::new(&t, 2).unwrap();
    s.find_neighbours_of_id(&t, 3).unwrap();
    assert_eq!(s.neighbours(), vec![2, 1]);
    assert_delta!(s.distance_k(0), 221.0);
    assert_delta!(s.distance_k(1), 544.0);

    Ok(())
}

#[test]
fn searcher_weighted() -> Result<(), NodeError> {
    let m = matrix_new();
    let t = TreeBuilder::new(&m)
        .try_bucket_size(2)?
        .build(&mut [0, 1, 2, 3, 4])?;
    let p = Probabilities::new(5, 0.25).unwrap();

    let mut s = SearcherWeighted::new(&t).unwrap();
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
