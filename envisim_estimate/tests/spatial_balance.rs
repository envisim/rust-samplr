use envisim_estimate::spatial_balance::*;
use envisim_test_utils::*;
use envisim_utils::matrix::Matrix;
use envisim_utils::sampling_options::SamplingOptions;

#[test]
fn test_voronoi() {
    let data = Matrix::new(&DATA_10_2, 10).unwrap();
    let options = SamplingOptions::new(&PROB_10_E)
        .unwrap()
        .set_spreading(data)
        .unwrap();

    let sb = voronoi(&[0], &options).unwrap();
    assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2));
}

#[test]
fn test_local() {
    let data = Matrix::new(&DATA_10_2, 10).unwrap();
    let options = SamplingOptions::new(&PROB_10_E)
        .unwrap()
        .set_spreading(data)
        .unwrap();

    let sb = local(&[0], &options, true).unwrap();
    assert_delta!(sb, 0.9734661634680257247254);

    let sb = local(&[0, 1], &options, true).unwrap();
    assert_delta!(sb, 1.251849435249984709984);
}
