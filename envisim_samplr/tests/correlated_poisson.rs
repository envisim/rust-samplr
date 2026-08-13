//! Test cp-sampling

mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_cps() {
    let options = Data10::options_u();
    test_wor(|rng| options.cps(rng), &options, 1e-2, 100000);
}

#[test]
fn test_scps() {
    let options = Data10::options_u();
    test_wor(|rng| options.scps(rng), &options, 1e-2, 100000);
    let options = Data10::options_e();
    test_wor(|rng| options.scps(rng), &options, 1e-2, 100000);
}

#[test]
fn test_lcps() {
    let options = Data10::options_u();
    test_wor(|rng| options.lcps(rng), &options, 1e-2, 10000);
}
