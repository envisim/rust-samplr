use envisim_estimate::balance::*;
use envisim_samplr::{SampleOptions, SamplingError};
use envisim_test_utils::*;
use envisim_utils::Matrix;

#[test]
fn test_balance() -> Result<(), SamplingError> {
    let data = Matrix::new(&DATA_10_2, 10);
    let mut options = SampleOptions::new(&PROB_10_E)?.set_spreading(&data)?;

    let sb = balance_deviation(&[0], &options)?;
    let dev: Vec<f64> = vec![
        data.col_iter(0).sum::<f64>() - data[(0, 0)] / PROB_10_E[0],
        data.col_iter(1).sum::<f64>() - data[(0, 1)] / PROB_10_E[0],
    ];

    assert_fvec(sb.0.as_ref().unwrap(), &dev);
    assert_eq!(sb.1, None);

    options = options.set_balancing(&data)?;
    let sb = balance_deviation(&[0], &options)?;

    assert_fvec(sb.0.as_ref().unwrap(), &dev);
    assert_fvec(sb.1.as_ref().unwrap(), &dev);

    Ok(())
}
