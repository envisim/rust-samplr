use envisim_samplr::systematic::*;
use envisim_test_utils::*;

mod test_utils;
use test_utils::*;

#[test]
fn systematic() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(sample, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn srs_wr() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(sample_random_order, &mut rng, &opts, p, 1e-2, 100000)
}
