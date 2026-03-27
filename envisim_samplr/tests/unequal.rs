mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_sampford() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || options.sampford(&mut rng).unwrap(),
        &options,
        1e-2,
        10000,
    );
}

#[test]
fn test_pareto() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.pareto(&mut rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_brewer() {
    let mut rng = rng();
    let options = options_unequal_e();
    test_wor(|| options.brewer(&mut rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_poisson() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.poisson(&mut rng), &options, 1e-2, 100000);
}

// So inefficient...
#[test]
fn test_conditional_poisson() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(
        || options.conditional_poisson(&mut rng, 5).unwrap(),
        &options,
        1e-1,
        100000,
    );
}
