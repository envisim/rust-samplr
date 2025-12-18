mod test_utils;
use envisim_samplr::systematic::{
    systematic,
    systematic_random_order,
};
use test_utils::*;

#[test]
fn test_systematic() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| systematic(&mut rng, &options), &options, 1e-2, 100000);
}

#[test]
fn test_systematic_random() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || systematic_random_order(&mut rng, &options),
        &options,
        1e-2,
        100000,
    );
}
