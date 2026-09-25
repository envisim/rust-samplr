//! Test cube sampling

mod test_utils;
use envisim_samplr::cube_method::*;
use test_utils::*;

#[test]
fn test_cube() {
    let (spec, bmat) = matrix_big_balanced();
    let options = SamplingOptions::with_spec(spec);
    test_wor(|rng| options.cube(rng, &bmat).unwrap(), &options, 1e-0, 100);

    let options = Data10::options_u();
    let bmat = Data10::bmatrix_up();
    test_wor(
        |rng| options.cube(rng, &bmat).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_sequential_cube() {
    let (spec, bmat) = matrix_big_balanced();
    let options = SamplingOptions::with_spec(spec);
    test_wor(
        |rng| options.sequential_cube(rng, &bmat).unwrap(),
        &options,
        1e-0,
        100,
    );

    let options = Data10::options_u();
    let bmat = Data10::bmatrix_up();
    test_wor(
        |rng| options.sequential_cube(rng, &bmat).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_lcube() {
    let options = Data10::options_u();
    let bal = Data10::bmatrix_up();
    test_wor(
        |rng| options.local_cube(rng, &bal).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_cube_stratified() {
    let options = Data10::options_e();
    let bal = Data10::bmatrix_ep();
    test_wor(
        |rng| {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = cube_stratified(rng, &options, &bal, &grps).unwrap();
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
    let options = Data10::options_e();
    let bal = Data10::bmatrix_ep();
    test_wor(
        |rng| {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = local_cube_stratified(rng, &options, &bal, &grps).unwrap();
            s
        },
        &options,
        0.015,
        100000,
    );
}
