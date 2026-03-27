mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_cps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.cps(&mut rng), &options, 1e-2, 100000);
}

#[test]
fn test_scps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.scps(&mut rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_lcps() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.lcps(&mut rng).unwrap(), &options, 1e-2, 10000);
}
