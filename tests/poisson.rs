use envisim_samplr::poisson::*;
use envisim_test_utils::*;
use envisim_utils::matrix::RefMatrix;

mod test_utils;
use test_utils::*;

#[test]
fn poisson() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| sample(&mut rng, p), p, 1e-2, 100000);
}

// So inefficient...
#[test]
fn conditional() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| conditional_sample(&mut rng, p, 5, 1000), p, 1e-1, 100000);
}

#[test]
fn cps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(
        || SequentialCorrelatedPoissonSampling::sample(&mut rng, p, EPS),
        p,
        1e-2,
        100000,
    );
}

#[test]
fn scps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(
        || SpatiallyCorrelatedPoissonSampling::sample(&mut rng, p, EPS, &data, NONZERO_2),
        p,
        1e-2,
        100000,
    );
}

#[test]
fn lcps() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(
        || LocallyCorrelatedPoissonSampling::sample(&mut rng, p, EPS, &data, NONZERO_2),
        p,
        1e-2,
        100000,
    );
}
