use envisim_estimate::balance::*;
use envisim_test_utils::*;
use envisim_utils::matrix::Matrix;
use envisim_utils::sampling_options::SamplingOptions;

#[test]
fn test_balance() {
    let data = Matrix::new(&DATA_10_2, 10).unwrap();
    let options = SamplingOptions::new(&PROB_10_E)
        .unwrap()
        .set_spreading(data.clone_shallow())
        .unwrap();

    let sb = balance_deviation_spreading(&[0], &options).unwrap();
    let dev = vec![
        data.col_iter(0).sum::<f64>() - data[(0, 0)] / PROB_10_E[0],
        data.col_iter(1).sum::<f64>() - data[(0, 1)] / PROB_10_E[0],
    ];

    assert_fvec(&sb, &dev);

    let options = options.set_balancing(data.clone_shallow()).unwrap();
    let sb = balance_deviation_balancing(&[0], &options).unwrap();

    assert_fvec(&sb, &dev);
}
