use envisim_samplr::pivotal_method::{lpm_2, SampleOptions};
use envisim_utils::Matrix;
use extendr_api::prelude::*;
use extendr_api::wrapper::matrix::RMatrix;
use rand::{rngs::SmallRng, SeedableRng};

#[extendr]
fn rust_lpm_2(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());

    SampleOptions::new(r_prob)
        .unwrap()
        .auxiliaries(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap()
        .eps(r_eps)
        .unwrap()
        .sample(&mut rng, lpm_2)
        .unwrap()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod samplr;
    fn rust_lpm_2;
}
