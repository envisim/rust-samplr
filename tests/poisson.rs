use envisim_samplr::poisson::*;
use envisim_test_utils::*;
use envisim_utils::Matrix;

mod test_utils;
use test_utils::*;

#[test]
fn test_poisson() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(sample, &mut rng, &opts, p, 1e-2, 100000)
}

// So inefficient...
#[test]
fn test_conditional() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor2(|| conditional(&mut rng, &opts, 5), p, 1e-1, 100000)
}

#[test]
fn test_cps() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(cps, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_scps() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = Matrix::from_ref(&DATA_10_2, 10);
    let mut opts = SampleOptions::new(p)?;
    opts.auxiliaries(&data)?;

    test_wor(scps, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_lcps() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = Matrix::from_ref(&DATA_10_2, 10);
    let mut opts = SampleOptions::new(p)?;
    opts.auxiliaries(&data)?;

    test_wor(lcps, &mut rng, &opts, p, 1e-2, 100000)
}
