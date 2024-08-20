use envisim_test_utils::*;
use envisim_utils::probability::Probabilities;

fn prob_new() -> Probabilities {
    Probabilities::with_values(&vec![0.1, 0.2, 0.0, 1.0, 0.6, 0.8]).unwrap()
}

#[test]
fn check() {
    assert!(Probabilities::new(2, 0.1).is_ok());
    assert!(Probabilities::new(2, 0.9).is_ok());
    assert!(Probabilities::new(2, -0.9).is_err());
    assert!(Probabilities::new(2, 1.9).is_err());
    assert!(Probabilities::new(2, f64::NAN).is_err());
    assert!(Probabilities::check(&vec![0.1, 0.2]).is_ok());
    assert!(Probabilities::check(&vec![0.1, -0.2]).is_err());
    assert!(Probabilities::check(&vec![0.1, 1.2]).is_err());
    assert!(Probabilities::check(&vec![0.1, f64::NAN]).is_err());

    assert!(Probabilities::check_eps(EPS).is_ok());
    assert!(Probabilities::check_eps(-0.1).is_err());
    assert!(Probabilities::check_eps(1.0).is_err());
}

#[test]
fn is_zero() {
    let mut p = prob_new();
    assert!(!p.is_zero(0));
    assert!(p.is_zero(2));
    assert!(!p.is_one(0));
    assert!(p.is_one(3));

    p.eps = 1e-2;
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
