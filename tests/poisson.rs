use envisim_samplr::poisson::*;
use envisim_test_utils::*;
use envisim_utils::matrix::RefMatrix;

mod test_utils;
use test_utils::*;

#[test]
fn test_poisson() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| sample(&mut rng, p), p, 1e-2, 100000);
}

// So inefficient...
#[test]
fn test_conditional() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| conditional(&mut rng, p, 5, 1000), p, 1e-1, 100000);
}

#[test]
fn test_cps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| cps(&mut rng, p, EPS), p, 1e-2, 100000);
}

#[test]
fn test_scps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(|| scps(&mut rng, p, EPS, &data, NONZERO_2), p, 1e-2, 100000);
}

#[test]
fn test_lcps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(|| lcps(&mut rng, p, EPS, &data, NONZERO_2), p, 1e-2, 100000);
}
