use envisim_samplr::correlated_poisson::*;
use envisim_test_utils::*;
use envisim_utils::{random::*, Matrix};

mod test_utils;
use test_utils::*;

#[test]
fn test_cps() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(cps, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_scps() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_U;
    let data = Matrix::new(&DATA_10_2, 10).unwrap();
    let opts = SampleOptions::new(p)?.set_spreading(&data)?;

    test_wor(scps, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_lcps() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_U;
    let data = Matrix::new(&DATA_10_2, 10).unwrap();
    let opts = SampleOptions::new(p)?.set_spreading(&data)?;

    test_wor(lcps, &mut rng, &opts, p, 1e-2, 100000)
}
