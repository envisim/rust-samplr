mod test_utils;
use envisim_samplr::*;
use test_utils::*;

#[test]
fn test_sampford() {
    let options = Data10::options_u();
    test_wor(|rng| options.sampford(rng).unwrap(), &options, 1e-2, 10000);
}

#[test]
fn test_pareto() {
    let options = Data10::options_u();
    test_wor(|rng| options.pareto(rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_brewer() {
    let options = Data10::options_u();
    test_wor(|rng| options.brewer(rng).unwrap(), &options, 1e-2, 100000);
}

#[test]
fn test_poisson() {
    let options = Data10::options_u();
    test_wor_random_n(|rng| options.poisson(rng), &options, 1e-2, 100000);
}

// So inefficient...
#[test]
fn test_conditional_poisson() {
    let options = Data10::options_u();
    test_wor(
        |rng| options.conditional_poisson(rng, 5).unwrap(),
        &options,
        1e-1,
        100000,
    );
}
