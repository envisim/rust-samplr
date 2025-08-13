use envisim_estimate::balance::balance_deviation;
use envisim_estimate::horvitz_thompson::local_mean_variance;
use envisim_estimate::spatial_balance::{local as sb_local, voronoi as sb_voronoi};
use envisim_samplr::correlated_poisson::{cps, lcps, scps};
use envisim_samplr::cube_method::{cube, cube_stratified, local_cube, local_cube_stratified};
use envisim_samplr::pivotal_method::{hierarchical_lpm_2, lpm_1, lpm_1s, lpm_2, rpm, spm};
use envisim_samplr::systematic::{
    sample as systematic, sample_random_order as systematic_random_order,
};
use envisim_samplr::unequal::{brewer, conditional_poisson, pareto, poisson, sampford};
use envisim_samplr::{AuxiliariesOptions, SampleOptions};
use envisim_utils::pips::pips_from_slice;
use envisim_utils::Matrix;
use extendr_api::prelude::*;
use extendr_api::wrapper::matrix::RMatrix;
use rand::{rngs::SmallRng, SeedableRng};
use std::num::NonZeroUsize;

#[extendr]
fn rust_unequal(
    r_prob: &[f64],
    r_eps: f64,
    r_seed: u64,
    r_method: &str,
    r_max_iter: usize,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let max_iter = NonZeroUsize::new(r_max_iter).unwrap();

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_max_iterations(max_iter)
        .unwrap();

    let s = match r_method {
        "spm" => spm(&mut rng, &options),
        "cps" => cps(&mut rng, &options),
        "poisson" => options.sample(&mut rng, poisson),
        "systematic" => systematic(&mut rng, &options),
        "systematic_random_order" => systematic_random_order(&mut rng, &options),
        "brewer" => brewer(&mut rng, &options),
        "pareto" => pareto(&mut rng, &options),
        "sampford" => sampford(&mut rng, &options),
        "rpm" | &_ => rpm(&mut rng, &options),
    };

    s.unwrap()
}

#[extendr]
fn rust_unequal_conditional_poisson(
    r_prob: &[f64],
    r_sample_size: usize,
    r_eps: f64,
    r_seed: u64,
    r_max_iter: usize,
) -> Vec<usize> {
    let mut rng = SmallRng::seed_from_u64(r_seed);
    let max_iter = NonZeroUsize::new(r_max_iter).unwrap();

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_max_iterations(max_iter)
        .unwrap();

    conditional_poisson(&mut rng, &options, r_sample_size).unwrap()
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

    let aux = AuxiliariesOptions::new(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap();
    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_spreading_options(aux)
        .unwrap();

    let s = match r_method {
        "lpm_1" => lpm_1(&mut rng, &options),
        "lpm_1s" => lpm_1s(&mut rng, &options),
        "scps" => scps(&mut rng, &options),
        "lcps" => lcps(&mut rng, &options),
        "lpm_2" | &_ => lpm_2(&mut rng, &options),
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

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_balancing(&bal_data)
        .unwrap();

    let s = match r_method {
        "cube" | &_ => cube(&mut rng, &options),
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

    let aux = AuxiliariesOptions::new(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap();
    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_balancing(&bal_data)
        .unwrap()
        .set_spreading_options(aux)
        .unwrap();

    let s = match r_method {
        "local_cube" | &_ => local_cube(&mut rng, &options),
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

    let aux = AuxiliariesOptions::new(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap();
    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_spreading_options(aux)
        .unwrap();

    let sizes: Vec<usize> = r_sizes
        .iter()
        .map(|&x| usize::try_from(x).unwrap_or(0))
        .collect();

    let s = match r_method {
        "lpm_2" | &_ => hierarchical_lpm_2(&mut rng, &options, &sizes),
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
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_balancing(&bal_data)
        .unwrap();

    let s = match r_method {
        "cube" | &_ => cube_stratified(&mut rng, &options, &strata),
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
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let aux = AuxiliariesOptions::new(&data)
        .unwrap()
        .try_bucket_size(r_bucket_size)
        .unwrap();
    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_eps(r_eps)
        .unwrap()
        .set_balancing(&bal_data)
        .unwrap()
        .set_spreading_options(aux)
        .unwrap();

    let s = match r_method {
        "local_cube" | &_ => local_cube_stratified(&mut rng, &options, &strata),
    };

    s.unwrap()
}

#[extendr]
fn rust_local_mean_variance(
    r_values: &[f64],
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_neighbours: usize,
) -> f64 {
    if r_neighbours == 0 {
        return f64::NAN;
    }

    let neighbours = NonZeroUsize::new(r_neighbours).unwrap();
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());

    let aux = AuxiliariesOptions::new(&data)
        .unwrap()
        .est_bucket_size()
        .unwrap();
    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_spreading_options(aux)
        .unwrap();

    local_mean_variance(r_values, &options, neighbours).unwrap()
}

#[extendr]
fn rust_spatial_balance_measure(
    r_sample: &[i32],
    r_prob: &[f64],
    r_data: RMatrix<f64>,
    r_method: &str,
) -> f64 {
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());
    let sample: Vec<usize> = r_sample.iter().map(|&x| x as usize).collect();

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_spreading(&data)
        .unwrap();

    let v = match r_method {
        "local" => sb_local(&sample, &options),
        "voronoi" | &_ => sb_voronoi(&sample, &options),
    };

    v.unwrap_or(-1.0)
}

#[extendr]
fn rust_balance_deviation(r_sample: &[i32], r_prob: &[f64], r_data: RMatrix<f64>) -> Vec<f64> {
    let data = Matrix::from_ref(r_data.data(), r_data.nrows());
    let sample: Vec<usize> = r_sample.iter().map(|&x| x as usize).collect();

    let options = SampleOptions::new(r_prob)
        .unwrap()
        .set_spreading(&data)
        .unwrap();

    balance_deviation(&sample, &options).unwrap().0.unwrap()
}

#[extendr]
fn rust_pips_from_values(r_values: &[f64], r_sample_size: usize) -> Vec<f64> {
    pips_from_slice(r_values, r_sample_size)
        .unwrap()
        .data()
        .to_vec()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod samplr;
    fn rust_unequal;
    fn rust_unequal_conditional_poisson;
    fn rust_spatially_balanced;
    fn rust_balanced;
    fn rust_doubly_balanced;
    fn rust_spatially_balanced_hierarchical;
    fn rust_balanced_stratified;
    fn rust_doubly_balanced_stratified;
    fn rust_local_mean_variance;
    fn rust_spatial_balance_measure;
    fn rust_balance_deviation;
    fn rust_pips_from_values;
}
