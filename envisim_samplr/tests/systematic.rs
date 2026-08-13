//! Test systematic sampling

mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_systematic() {
    let options = Data10::options_u();
    test_wor(|rng| options.systematic(rng), &options, 1e-2, 100000);
}

#[test]
fn test_systematic_random() {
    let options = Data10::options_u();
    test_wor(
        |rng| options.systematic_random_order(rng),
        &options,
        1e-2,
        100000,
    );
}
