use envisim_samplr::*;

mod test_utils;
use test_utils::*;

#[test]
fn srs_wor() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(|| options.srs(&mut rng), &options, 1e-2, 100000);
}

#[test]
fn srs_wr() {
    let mut rng = rng();
    let options = options_equal_10_2();
    test_wor(
        || options.srs_with_replacement(&mut rng),
        &options,
        1e-2,
        100000,
    );
}
