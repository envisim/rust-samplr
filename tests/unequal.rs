use envisim_samplr::unequal::*;
use envisim_test_utils::*;
use envisim_utils::random::*;

mod test_utils;
use test_utils::*;

#[test]
fn test_sampford() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor2(|| sampford(&mut rng, &opts), p, 1e-2, 10000)
}

#[test]
fn test_pareto() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor(pareto, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_brewer() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_E;
    let opts = SampleOptions::new(p)?;

    test_wor(brewer, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_poisson() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor(poisson, &mut rng, &opts, p, 1e-2, 100000)
}

// So inefficient...
#[test]
fn test_conditional_poisson() -> Result<(), SamplingError> {
    let mut rng = SmallRng::seed_from_u64(42);
    let p = &PROB_10_U;
    let opts = SampleOptions::new(p)?;

    test_wor2(|| conditional_poisson(&mut rng, &opts, 5), p, 1e-1, 100000)
}
