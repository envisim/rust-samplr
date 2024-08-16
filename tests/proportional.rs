use envisim_samplr::proportional::*;
use envisim_test_utils::*;

mod test_utils;
use test_utils::*;

#[test]
fn test_sampford() {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;

    test_wor(|| sampford(&mut rng, p, EPS, 1000), p, 1e-2, 10000);
}

#[test]
fn test_pareto() {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;

    test_wor(|| pareto(&mut rng, p, EPS), p, 1e-2, 100000);
}

#[test]
fn test_brewer() {
    let mut rng = seeded_rng();
    let p = &PROB_10_E;

    test_wor(|| brewer(&mut rng, p, EPS), p, 1e-2, 100000);
}
