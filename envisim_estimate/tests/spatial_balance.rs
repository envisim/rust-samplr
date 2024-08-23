use envisim_estimate::spatial_balance::*;
use envisim_test_utils::*;
use envisim_utils::Matrix;

#[test]
fn test_voronoi() -> Result<(), SamplingError> {
    let data = Matrix::new(&DATA_10_2, 10);
    let sb = voronoi(&[0], &PROB_10_E, &(&data).into())?;
    assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2));
    Ok(())
}

#[test]
fn test_local() -> Result<(), SamplingError> {
    let data = Matrix::new(&DATA_10_2, 10);
    let sb = local(&[0], &PROB_10_E, &(&data).into())?;
    assert_delta!(sb, 0.9734661634680257247254);
    let sb = local(&[0, 1], &PROB_10_E, &(&data).into())?;
    assert_delta!(sb, 1.251849435249984709984);
    Ok(())
}
