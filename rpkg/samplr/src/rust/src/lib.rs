use envisim_samplr::cube_method::{cube, cube_stratified, local_cube, local_cube_stratified};
use envisim_samplr::pivotal_method::{hierarchical_lpm_2, lpm_1, lpm_1s, lpm_2, rpm, spm};
use envisim_samplr::poisson::{
    conditional as conditional_poisson, cps, lcps, sample as poisson, scps,
};
use envisim_samplr::systematic::{
    sample as systematic, sample_random_order as systematic_random_order,
};
use envisim_samplr::unequal::{brewer, pareto, sampford};
use envisim_samplr::SampleOptions;
use envisim_utils::Matrix;
use extendr_api::prelude::*;
use extendr_api::wrapper::matrix::RMatrix;
use rand::{rngs::SmallRng, SeedableRng};

#[extendr]
fn rust_simple_unequal(
    r_prob: &[f64],
    r_eps: f64,
    r_seed: u64,
    r_method: &str,
    r_max_iter: usize,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);

    let mut options = SampleOptions::new(r_prob).unwrap();
    options.eps(r_eps).unwrap();

    let s = match r_method {
        "rpm" => options.sample(&mut rng, rpm),
        "spm" => options.sample(&mut rng, spm),
        "cps" => options.sample(&mut rng, cps),
        "poisson" => options.sample(&mut rng, poisson),
        "conditional_poisson" => conditional_poisson(&mut rng, &options, r_max_iter),
        "systematic" => options.sample(&mut rng, systematic),
        "systematic_random_order" => options.sample(&mut rng, systematic_random_order),
        "brewer" => options.sample(&mut rng, brewer),
        "pareto" => options.sample(&mut rng, pareto),
        "sampford" => options.sample(&mut rng, sampford),
        &_ => options.sample(&mut rng, rpm),
    };

    s.unwrap()
}

#[extendr]
fn rust_spatially_balanced(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
    r_method: &str,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());

    let mut options = SampleOptions::new(r_prob).unwrap();
    options
        .auxiliaries(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap()
        .eps(r_eps)
        .unwrap();

    let s = match r_method {
        "lpm_1" => options.sample(&mut rng, lpm_1),
        "lpm_1s" => options.sample(&mut rng, lpm_1s),
        "lpm_2" => options.sample(&mut rng, lpm_2),
        "scps" => options.sample(&mut rng, scps),
        "lcps" => options.sample(&mut rng, lcps),
        &_ => options.sample(&mut rng, lpm_2),
    };

    s.unwrap()
}

#[extendr]
fn rust_balanced(
    r_prob: &[f64],
    r_bal_data: RMatrix<f64>,
    r_eps: f64,
    r_seed: u64,
    r_method: &str,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let bal_data = Matrix::from_ref(r_bal_data.data(), r_bal_data.nrows());

    let mut options = SampleOptions::new(r_prob).unwrap();
    options.balancing(&bal_data).unwrap().eps(r_eps).unwrap();

    let s = match r_method {
        "cube" => options.sample(&mut rng, cube),
        &_ => options.sample(&mut rng, cube),
    };

    s.unwrap()
}

#[extendr]
fn rust_doubly_balanced(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_bal_data: RMatrix<f64>,
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
    r_method: &str,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());
    let bal_data = Matrix::from_ref(r_bal_data.data(), r_bal_data.nrows());

    let mut options = SampleOptions::new(r_prob).unwrap();
    options
        .balancing(&bal_data)
        .unwrap()
        .auxiliaries(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap()
        .eps(r_eps)
        .unwrap();

    let s = match r_method {
        "local_cube" => options.sample(&mut rng, local_cube),
        &_ => options.sample(&mut rng, local_cube),
    };

    s.unwrap()
}

#[extendr]
fn rust_spatially_balanced_hierarchical(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_sizes: &[i32],
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
    r_method: &str,
) -> RMatrix<i32> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());

    let mut options = SampleOptions::new(r_prob).unwrap();
    options
        .auxiliaries(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap()
        .eps(r_eps)
        .unwrap();

    let sizes: Vec<usize> = r_sizes
        .iter()
        .map(|&x| usize::try_from(x).unwrap_or(0))
        .collect();

    let s = match r_method {
        "lpm_2" => hierarchical_lpm_2(&mut rng, &options, &sizes),
        &_ => hierarchical_lpm_2(&mut rng, &options, &sizes),
    };

    let n = sizes.iter().sum();
    let mut return_matrix = RMatrix::<i32>::new(n, 2);

    let mut idx: usize = 0;
    for (i, vec) in s.unwrap().iter().enumerate() {
        for &j in vec.iter() {
            return_matrix[[idx, 0usize]] = i32::try_from(j).unwrap_or(-1);
            return_matrix[[idx, 1usize]] = i32::try_from(i).unwrap_or(-1);
            idx += 1;
        }
    }

    return_matrix
}

#[extendr]
fn rust_balanced_stratified(
    r_prob: &[f64],
    r_bal_data: RMatrix<f64>,
    r_strata: &[i32],
    r_eps: f64,
    r_seed: u64,
    r_method: &str,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let bal_data = Matrix::from_ref(r_bal_data.data(), r_bal_data.nrows());
    let strata: Vec<i64> = r_strata.iter().map(|&x| x as i64).collect();

    let mut options = SampleOptions::new(r_prob).unwrap();
    options.balancing(&bal_data).unwrap().eps(r_eps).unwrap();

    let s = match r_method {
        "cube" => cube_stratified(&mut rng, &options, &strata),
        &_ => cube_stratified(&mut rng, &options, &strata),
    };

    s.unwrap()
}

#[extendr]
fn rust_doubly_balanced_stratified(
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_bal_data: RMatrix<f64>,
    r_strata: &[i32],
    r_eps: f64,
    r_bucket_size: usize,
    r_seed: u64,
    r_method: &str,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());
    let bal_data = Matrix::from_ref(r_bal_data.data(), r_bal_data.nrows());
    let strata: Vec<i64> = r_strata.iter().map(|&x| x as i64).collect();

    let mut options = SampleOptions::new(r_prob).unwrap();
    options
        .balancing(&bal_data)
        .unwrap()
        .auxiliaries(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap()
        .eps(r_eps)
        .unwrap();

    let s = match r_method {
        "local_cube" => local_cube_stratified(&mut rng, &options, &strata),
        &_ => local_cube_stratified(&mut rng, &options, &strata),
    };

    s.unwrap()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod samplr;
    fn rust_simple_unequal;
    fn rust_spatially_balanced;
    fn rust_balanced;
    fn rust_doubly_balanced;
    fn rust_spatially_balanced_hierarchical;
    fn rust_balanced_stratified;
    fn rust_doubly_balanced_stratified;
}
