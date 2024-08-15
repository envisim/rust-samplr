use envisim_estimate::spatial_balance::*;
use envisim_test_utils::*;
use envisim_utils::matrix::RefMatrix;

#[test]
fn test_voronoi() {
    let data = RefMatrix::new(&DATA_10_2, 10);
    let sb = voronoi(&PROB_10_E, &data, &[0], NONZERO_2).unwrap();
    assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2));
}

#[test]
fn test_local() {
    let data = RefMatrix::new(&DATA_10_2, 10);
    let sb = local(&PROB_10_E, &data, &[0], NONZERO_2).unwrap();
    assert_delta!(sb, 0.9734661634680257247254);
    let sb = local(&PROB_10_E, &data, &[0, 1], NONZERO_2).unwrap();
    assert_delta!(sb, 1.251849435249984709984);
}
