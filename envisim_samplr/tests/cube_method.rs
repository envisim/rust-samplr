mod test_utils;
use envisim_samplr::cube_method::*;
use test_utils::*;

#[test]
fn test_cube() {
    let (spec, bmat) = matrix_big_balanced();
    let options = SamplingOptions::with_spec(spec)
        .set_balancing(bmat)
        .unwrap();
    test_wor_fixed_n(|rng| options.cube(rng), &options, 100);

    let options = Data10::options_u();
    test_wor(|rng| options.cube(rng), &options, 1e-2, 100000);
}

#[test]
fn test_sequential_cube() {
    let (spec, bmat) = matrix_big_balanced();
    let options = SamplingOptions::with_spec(spec)
        .set_balancing(bmat)
        .unwrap();
    test_wor_fixed_n(|rng| options.sequential_cube(rng), &options, 100);

    let options = Data10::options_u();
    test_wor(|rng| options.sequential_cube(rng), &options, 1e-2, 100000);
}

#[test]
fn test_lcube() {
    let options = Data10::options_u();
    test_wor(|rng| options.local_cube(rng), &options, 1e-2, 100000);
}

#[test]
fn test_cube_stratified() {
    let options = Data10::options_e();
    test_wor(
        |rng| {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = cube_stratified(rng, &options, &grps).unwrap();
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
    test_wor(
        |rng| {
            let grps: [i64; 10] = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
            let s = local_cube_stratified(rng, &options, &grps).unwrap();
            s
        },
        &options,
        0.015,
        100000,
    );
}
