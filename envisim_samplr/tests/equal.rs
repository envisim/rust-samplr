//! Test equal prob sampling

use envisim_samplr::*;

mod test_utils;
use test_utils::*;

#[test]
fn srs_wor() {
    let options = Data10::options_e();
    test_wor(|rng| options.srs(rng), &options, 1e-2, 100000);
}

#[test]
fn srs_wr() {
    let options = Data10::options_e();
    test_wor(
        |rng| options.srs_with_replacement(rng),
        &options,
        1e-2,
        100000,
    );
}
