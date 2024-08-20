use envisim_samplr::pivotal_method::lpm_2;
use envisim_utils::matrix::RefMatrix;
use extendr_api::prelude::*;
use extendr_api::wrapper::matrix::RMatrix;
use rand::{rngs::SmallRng, SeedableRng};
use std::num::NonZeroUsize;

#[extendr]
fn rust_lpm_2(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = RefMatrix::new(r_data.data(), r_data.nrows());
    let bucket_size = NonZeroUsize::new(r_bucket_size).unwrap();

    lpm_2(&mut rng, r_prob, r_eps, &data, bucket_size).unwrap()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod samplr;
    fn rust_lpm_2;
}
