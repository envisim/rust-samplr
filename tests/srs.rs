use envisim_samplr::srs::*;
use envisim_test_utils::*;

mod test_utils;
use test_utils::*;

#[test]
fn srs_wor() {
    let mut rng = seeded_rng();

    test_wor(|| sample(&mut rng, 2, 10), &PROB_10_E, 1e-2, 100000);
}

#[test]
fn srs_wr() {
    let mut rng = seeded_rng();

    test_wor(
        || sample_with_replacement(&mut rng, 2, 10),
        &PROB_10_E,
        1e-2,
        100000,
    );
}
