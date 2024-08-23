use envisim_samplr::unequal::*;
use envisim_test_utils::*;

mod test_utils;
use test_utils::*;

#[test]
fn test_sampford() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor2(|| sampford(&mut rng, &opts, 1000), p, 1e-2, 10000)
}

#[test]
fn test_pareto() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor(pareto, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_brewer() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor(brewer, &mut rng, &opts, p, 1e-2, 100000)
}
