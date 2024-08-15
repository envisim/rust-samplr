use envisim_test_utils::*;
use envisim_utils::matrix::*;

const DATA_4_2: [f64; 8] = [
    0.0, 1.0, 2.0, 3.0, //
    10.0, 11.0, 12.0, 13.0, //
];

fn matrix_new<'a>() -> (Matrix, RefMatrix<'a>) {
    (Matrix::new(&DATA_4_2, 4), RefMatrix::new(&DATA_4_2, 4))
}

#[test]
fn operate_matrix() {
    let (mm, rm) = matrix_new();
    assert_eq!(mm.data(), DATA_4_2);
    assert_eq!(rm.data(), DATA_4_2);

    assert_eq!(mm.nrow(), 4);
    assert_eq!(rm.nrow(), 4);
    assert_eq!(mm.ncol(), 2);
    assert_eq!(rm.ncol(), 2);
    assert_eq!(mm.dim(), (4, 2));
    assert_eq!(rm.dim(), (4, 2));

    assert_eq!(
        mm.row_iter(0).cloned().collect::<Vec<f64>>(),
        vec![0.0, 10.0]
    );
    assert_eq!(
        rm.row_iter(1).cloned().collect::<Vec<f64>>(),
        vec![1.0, 11.0]
    );
    assert_eq!(
        mm.col_iter(0).cloned().collect::<Vec<f64>>(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        rm.col_iter(1).cloned().collect::<Vec<f64>>(),
        vec![10.0, 11.0, 12.0, 13.0]
    );

    assert_eq!(unsafe { mm.get_unchecked((0, 0)) }, &0.0);
    assert_eq!(unsafe { rm.get_unchecked((1, 1)) }, &11.0);
}

#[test]
fn distance_to_row() {
    let (mm, rm) = matrix_new();
    assert_eq!(mm.distance_to_row(0, &vec![10.0, 10.0]), 100.0 + 0.0);
    assert_eq!(rm.distance_to_row(1, &vec![10.0, 10.0]), 81.0 + 1.0);
}

#[test]
fn prod_vec() {
    let (mm, rm) = matrix_new();
    assert_eq!(
        mm.prod_vec(&vec![2.0, 3.0]),
        vec![30.0, 33.0 + 2.0, 36.0 + 4.0, 39.0 + 6.0]
    );
    assert_eq!(
        rm.prod_vec(&vec![1.0, 2.0]),
        vec![20.0, 22.0 + 1.0, 24.0 + 2.0, 26.0 + 3.0]
    );
}

#[test]
fn mult() {
    let (mm, rm) = matrix_new();
    let one_mat = Matrix::new_fill(1.0, (2, 4));
    let two_mat = Matrix::new_fill(2.0, (2, 4));
    assert_eq!(one_mat.mult(&mm).data(), vec![6.0, 6.0, 46.0, 46.0]);
    assert_eq!(two_mat.mult(&rm).data(), vec![12.0, 12.0, 92.0, 92.0]);
}

#[test]
fn resize() {
    let (mut mm, _) = matrix_new();
    mm.resize(2, 2);
    assert_eq!(mm.dim(), (2, 2));
}

#[test]
fn rref() {
    let mut data1 = Matrix::new(
        &[
            0.81, 0.46, 0.40, //
            0.54, 0.70, 0.08, //
            0.39, 0.42, 0.87, //
            0.64, 0.70, 0.32, //
        ],
        3,
    );

    data1.reduced_row_echelon_form();
    assert_fvec(&data1.data()[0..3], &vec![1.0, 0.0, 0.0]);
    assert_fvec(&data1.data()[3..6], &vec![0.0, 1.0, 0.0]);
    assert_fvec(&data1.data()[6..9], &vec![0.0, 0.0, 1.0]);
    assert_fvec(
        &data1.data()[9..12],
        &vec![0.188953701217875, 0.748566128914163, 0.212107159999675],
    );
}
