mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_systematic() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.systematic(&mut rng), &options, 1e-2, 100000);
}

#[test]
fn test_systematic_random() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || options.systematic_random_order(&mut rng),
        &options,
        1e-2,
        100000,
    );
}
