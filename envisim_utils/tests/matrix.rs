use envisim_test_utils::*;
use envisim_utils::matrix::{
    Matrix,
    MatrixIndex,
};

const DATA_4_2: [f64; 8] = [
    0.0, 1.0, 2.0, 3.0, //
    10.0, 11.0, 12.0, 13.0, //
];

fn matrix_new<'a>() -> Matrix<'a> { Matrix::new(&DATA_4_2, 4).unwrap() }

#[test]
fn borrow_vs_owned() {
    let mm = matrix_new();
    assert_eq!(mm.data(), DATA_4_2);
    let mut rm = mm.clone();
    rm.to_mut();
    assert_eq!(rm.data(), DATA_4_2);
    assert_eq!(mm.data(), rm.data());
}

#[test]
fn operate_matrix() {
    let mm = matrix_new();
    assert_eq!(mm.nrow(), 4);
    assert_eq!(mm.ncol(), 2);
    assert_eq!(mm.dims(), MatrixIndex::new((4, 2)));

    assert_eq!(
        mm.row_iter(0).cloned().collect::<Vec<f64>>(),
        vec![0.0, 10.0]
    );
    assert_eq!(
        mm.row_iter(1).cloned().collect::<Vec<f64>>(),
        vec![1.0, 11.0]
    );
    assert_eq!(
        mm.col_iter(0).cloned().collect::<Vec<f64>>(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        mm.col_iter(1).cloned().collect::<Vec<f64>>(),
        vec![10.0, 11.0, 12.0, 13.0]
    );
}

#[test]
fn distance_to_row() {
    let mm = matrix_new();
    assert_eq!(
        mm.distance_to_row(0, &vec![10.0, 10.0]).unwrap(),
        100.0 + 0.0
    );
    assert_eq!(
        mm.distance_to_row(1, &vec![10.0, 10.0]).unwrap(),
        81.0 + 1.0
    );
}

#[test]
fn prod_vec() {
    let mm = matrix_new();
    assert_eq!(
        mm.prod_vec(&vec![2.0, 3.0]).unwrap(),
        vec![30.0, 33.0 + 2.0, 36.0 + 4.0, 39.0 + 6.0]
    );
    assert_eq!(
        mm.prod_vec(&vec![1.0, 2.0]).unwrap(),
        vec![20.0, 22.0 + 1.0, 24.0 + 2.0, 26.0 + 3.0]
    );
}

#[test]
fn mult() {
    let mm = matrix_new();
    let one_mat = Matrix::from_value(1.0, (2, 4)).unwrap();
    let two_mat = Matrix::from_value(2.0, (2, 4)).unwrap();
    assert_eq!(
        one_mat.mult(&mm).unwrap().data(),
        vec![6.0, 6.0, 46.0, 46.0]
    );
    assert_eq!(
        two_mat.mult(&mm).unwrap().data(),
        vec![12.0, 12.0, 92.0, 92.0]
    );
}

#[test]
fn resize() {
    let mut mm = matrix_new();
    mm.resize((2, 2));
    assert_eq!(mm.dims(), MatrixIndex::new((2, 2)));
}

#[test]
fn rref() {
    let mut data1 = Matrix::from_vec(
        vec![
            0.81, 0.46, 0.40, //
            0.54, 0.70, 0.08, //
            0.39, 0.42, 0.87, //
            0.64, 0.70, 0.32, //
        ],
        3,
    )
    .unwrap();

    data1.reduced_row_echelon_form();
    assert_fvec(&data1.data()[0..3], &vec![1.0, 0.0, 0.0]);
    assert_fvec(&data1.data()[3..6], &vec![0.0, 1.0, 0.0]);
    assert_fvec(&data1.data()[6..9], &vec![0.0, 0.0, 1.0]);
    assert_fvec(
        &data1.data()[9..12],
        &vec![0.188953701217875, 0.748566128914163, 0.212107159999675],
    );
}
