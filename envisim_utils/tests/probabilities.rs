use envisim_test_utils::*;
use envisim_utils::probabilities::*;

static EPS: f64 = 1e-12;
static PROBABILITY_ARR: [f64; 6] = [0.1, 0.2, 0.0, 1.0, 0.6, 0.8];
fn prob_new(eps: f64) -> FloatProbabilities {
    FloatProbabilities::new(PROBABILITY_ARR.to_vec(), eps)
}

#[test]
fn is_zero() {
    let p = prob_new(EPS);
    assert!(!p.is_zero(0));
    assert!(p.is_zero(2));
    assert!(!p.is_max(0));
    assert!(p.is_max(3));

    let mut p = prob_new(1e-2);
    p.set(0, 0.999);
    assert!(p.is_max(0));
}

#[test]
fn weight() {
    let p = prob_new(EPS);
    assert_delta!(p.weight(0, 1), 0.2 / 0.9);
    assert_delta!(p.weight_to(0.1, 1), 0.2 / 0.9);
    assert_delta!(p.weight(4, 5), 0.2 / 0.6);
    assert_delta!(p.weight_to(0.6, 5), 0.2 / 0.6);
}
