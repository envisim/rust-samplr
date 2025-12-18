mod test_utils;
use envisim_samplr::unequal::{
    brewer,
    conditional_poisson,
    pareto,
    poisson,
    sampford,
};
use test_utils::*;

#[test]
fn test_sampford() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || sampford(&mut rng, &options).unwrap(),
        &options,
        1e-2,
        10000,
    );
}

#[test]
fn test_pareto() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || pareto(&mut rng, &options).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_brewer() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || brewer(&mut rng, &options).unwrap(),
        &options,
        1e-2,
        100000,
    );
}

#[test]
fn test_poisson() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| poisson(&mut rng, &options), &options, 1e-2, 100000);
}

// So inefficient...
#[test]
fn test_conditional_poisson() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || conditional_poisson(&mut rng, &options, 5).unwrap(),
        &options,
        1e-1,
        100000,
    );
}
