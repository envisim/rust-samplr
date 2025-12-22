use envisim_samplr::srs::*;

mod test_utils;
use test_utils::*;

#[test]
fn srs_wor() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(|| srs(&mut rng, &options), &options, 1e-2, 100000);
}

#[test]
fn srs_wr() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(
        || srs_with_replacement(&mut rng, &options),
        &options,
        1e-2,
        100000,
    );
}
