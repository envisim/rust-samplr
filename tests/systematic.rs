use envisim_samplr::systematic::*;
use envisim_test_utils::*;

mod test_utils;
use test_utils::*;

#[test]
fn systematic() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| sample(&mut rng, p), p, 1e-2, 100000);
}

#[test]
fn srs_wr() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| sample_random_order(&mut rng, p), p, 1e-2, 100000);
}
