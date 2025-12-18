mod test_utils;
use envisim_samplr::correlated_poisson::{
    cps,
    lcps,
    scps,
};
use test_utils::*;

#[test]
fn test_cps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| cps(&mut rng, &options), &options, 1e-2, 100000);
}

#[test]
fn test_scps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| scps(&mut rng, &options), &options, 1e-2, 100000);
}

#[test]
fn test_lcps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| lcps(&mut rng, &options), &options, 1e-2, 10000);
}
