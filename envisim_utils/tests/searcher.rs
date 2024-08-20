use envisim_test_utils::*;
use envisim_utils::kd_tree::*;

const DATA_4_2: [f64; 10] = [
    0.0, 1.0, 2.0, 13.0, 14.0, //
    0.0, 10.0, 20.0, 30.0, 40.0, //
];

fn matrix_new<'a>() -> envisim_utils::matrix::RefMatrix<'a> {
    envisim_utils::matrix::RefMatrix::new(&DATA_4_2, 5)
}

#[test]
fn searcher() {
    let m = matrix_new();
    let t = Node::with_midpoint_slide(NONZERO_2, &m, &mut vec![0, 1, 2, 3]).unwrap();

    let mut s = Searcher::new(&t, 1).unwrap();
    s.find_neighbours(&t, &vec![5.0, 5.0]).unwrap();
    assert_eq!(s.neighbours(), vec![1]);
    assert_delta!(s.distance_k(0), 41.0);

    let mut s = Searcher::new(&t, 2).unwrap();
    s.find_neighbours_of_id(&t, 3).unwrap();
    assert_eq!(s.neighbours(), vec![2, 1]);
    assert_delta!(s.distance_k(0), 221.0);
    assert_delta!(s.distance_k(1), 544.0);
}

#[test]
fn searcher_weighted() {
    let m = matrix_new();
    let t = Node::with_midpoint_slide(NONZERO_2, &m, &mut vec![0, 1, 2, 3, 4]).unwrap();
    let p = envisim_utils::probability::Probabilities::new(5, 0.25).unwrap();

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
}
