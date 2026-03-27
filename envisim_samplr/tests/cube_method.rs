mod test_utils;
use envisim_samplr::cube_method::*;
use envisim_utils::matrix::Matrix;
use test_utils::*;

// const BAL_DATA_10_1: [f64; 10] = [
//     0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
// ];
// const BAL_DATA_10_1_P: [f64; 20] = [
//     0.20f64, 0.25, 0.35, 0.40, 0.50, 0.50, 0.55, 0.65, 0.70, 0.90, //
//     0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
// ];

#[test]
fn test_cube() {
    let mut rng = rng();
    let (p_vec, b_vec) = matrix_big_balanced();
    let bal = Matrix::new(&b_vec, 1000).unwrap();
    let options = SamplingOptions::new(&p_vec)
        .unwrap()
        .set_balancing(bal)
        .unwrap();
    test_wor_fixed_n(|| options.cube(&mut rng).unwrap(), &options, 100);

    let options = options_unequal();
    test_wor(|| options.cube(&mut rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_lcube() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || options.local_cube(&mut rng).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_cube_stratified() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(
        || {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = cube_stratified(&mut rng, &options, &grps).unwrap();
            assert_eq!(s.len(), 2);
            assert!((0..5).contains(&s[0])); // first unit from first grp
            assert!((5..10).contains(&s[1])); // second unit from second grp
            s
        },
        &options,
        0.015,
        100000,
    );
}

#[test]
fn test_lcube_stratified() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(
        || {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = local_cube_stratified(&mut rng, &options, &grps).unwrap();
            s
        },
        &options,
        0.015,
        100000,
    );
}
