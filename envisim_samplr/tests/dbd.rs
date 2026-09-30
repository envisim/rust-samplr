//! Test dbd.

mod test_utils;
use envisim_samplr::dbd::*;
use envisim_utils::sampling_options::BaseProbabilitiesSpec;
use test_utils::*;

#[test]
fn test_dbd_circular() {
    let mut rng = rng();
    let options = Data10::options_e().set_max_iterations(nz(100)).unwrap();
    let dbd_opts = DistributionalDesignOptions::default();
    let design = options
        .dbd_circular(&mut rng, dbd_opts)
        .unwrap()
        .into_sequence();
    // All ids should be in circle
    assert!((0..10).all(|id| design.contains(&id)));
}

#[test]
fn test_dbd_tc() {
    let mut rng = rng();
    let options = Data10::options_e().set_max_iterations(nz(100)).unwrap();
    let dbd_opts = DistributionalDesignOptions::default();
    let design = options.dbd_tc(&mut rng, dbd_opts).unwrap();
    let buckets = design.buckets();
    // All inc probs should be respected
    assert!(
        (0..10).all(|id| buckets.data().iter().filter(|v| **v == id).count()
            == design.tcp().n_repeats().get())
    );
    // Sample size should be respected for entire design
    assert!(buckets.nrow().get() == options.probabilities().sample_size());
}
