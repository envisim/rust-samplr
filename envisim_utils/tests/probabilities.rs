use envisim_test_utils::*;
use envisim_utils::probabilities::*;
use envisim_utils::sampling_options::SamplingOptionsError;

static PROBABILITY_ARR: [f64; 6] = [0.1, 0.2, 0.0, 1.0, 0.6, 0.8];
fn prob_new() -> ProbabilitiesUnequal {
    ProbabilitiesUnequal::with_values_unchecked(PROBABILITY_ARR.to_vec(), 1e-12)
}

#[test]
fn check() -> Result<(), SamplingOptionsError> {
    ProbabilitiesUnequal::with_value(2, 0.1, 0.0)?;
    ProbabilitiesUnequal::with_value(2, 0.9, 0.0)?;
    assert!(ProbabilitiesUnequal::with_value(2, -0.9, 0.0).is_err());
    assert!(ProbabilitiesUnequal::with_value(2, 1.9, 0.0).is_err());
    assert!(ProbabilitiesUnequal::with_value(2, f64::NAN, 0.0).is_err());

    Ok(())
}

#[test]
fn is_zero() {
    let mut p = prob_new();
    assert!(!p.is_zero(0));
    assert!(p.is_zero(2));
    assert!(!p.is_one(0));
    assert!(p.is_one(3));

    p.set_eps(1e-2).unwrap();
    p[0] = 0.999;
    assert!(p.is_one(0));
}

#[test]
fn weight() {
    let p = prob_new();
    assert_delta!(p.weight(0, 1), 0.2 / 0.9);
    assert_delta!(p.weight_to(0.1, 1), 0.2 / 0.9);
    assert_delta!(p.weight(4, 5), 0.2 / 0.6);
    assert_delta!(p.weight_to(0.6, 5), 0.2 / 0.6);
}
