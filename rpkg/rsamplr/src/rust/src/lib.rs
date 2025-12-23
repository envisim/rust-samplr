use envisim_estimate::balance::balance_deviation_spreading;
use envisim_estimate::horvitz_thompson::local_mean_variance;
use envisim_estimate::spatial_balance::{
    energy_distance as sb_energy,
    local as sb_local,
    voronoi as sb_voronoi,
};
use envisim_samplr::correlated_poisson::{
    cps,
    lcps,
    scps,
};
use envisim_samplr::cube_method::{
    cube,
    cube_stratified,
    local_cube,
    local_cube_stratified,
};
use envisim_samplr::pivotal_method::{
    hierarchical_lpm_2,
    lpm_1,
    lpm_1s,
    lpm_2,
    rpm,
    spm,
};
use envisim_samplr::systematic::{
    systematic,
    systematic_random_order,
};
use envisim_samplr::unequal::{
    brewer,
    conditional_poisson,
    pareto,
    poisson,
    sampford,
};
use envisim_utils::pips::{
    Probabilities,
    pips_from_slice,
};
use envisim_utils::sampling_options::{
    SamplingOptions,
    SpreadingOptions,
};
use savvy::{
    IntegerSexp,
    OwnedIntegerSexp,
    RealSexp,
    Sexp,
    savvy,
    savvy_err,
};

mod matrix;
mod random;
mod utils;

use matrix::*;
use random::*;
use utils::*;

#[savvy]
fn rust_unequal(
    r_prob: RealSexp,
    r_eps: f64,
    r_method: &str,
    r_max_iter: i32,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_max_iterations(i32_to_nonzerousize(r_max_iter)?)?;

    let s = match r_method {
        "spm" => spm(&mut rng, &options),
        "cps" => cps(&mut rng, &options),
        "poisson" => poisson(&mut rng, &options),
        "systematic" => systematic(&mut rng, &options),
        "systematic_random_order" => systematic_random_order(&mut rng, &options),
        "brewer" => brewer(&mut rng, &options)?,
        "pareto" => pareto(&mut rng, &options)?,
        "sampford" => sampford(&mut rng, &options)?,
        "rpm" | &_ => rpm(&mut rng, &options),
    };

    return_sample(s)
}

#[savvy]
fn rust_unequal_conditional_poisson(
    r_prob: RealSexp,
    r_sample_size: i32,
    r_eps: f64,
    r_max_iter: i32,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_max_iterations(i32_to_nonzerousize(r_max_iter)?)?;
    let sample_size = i32_to_usize(r_sample_size)?;

    let s = conditional_poisson(&mut rng, &options, sample_size)?;

    return_sample(s)
}

#[savvy]
fn rust_spatially_balanced(
    r_prob: RealSexp,
    r_data: RealSexp,
    r_eps: f64,
    r_bucket_size: i32,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let bucket_size = i32_to_nonzerousize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_spreading(aux)?;

    let s = match r_method {
        "lpm_1" => lpm_1(&mut rng, &options),
        "lpm_1s" => lpm_1s(&mut rng, &options),
        "scps" => scps(&mut rng, &options),
        "lcps" => lcps(&mut rng, &options),
        "lpm_2" | &_ => lpm_2(&mut rng, &options),
    };

    return_sample(s)
}

#[savvy]
fn rust_balanced(
    r_prob: RealSexp,
    r_bal_data: RealSexp,
    r_eps: f64,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);

    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?;

    let s = match r_method {
        "cube" | &_ => cube(&mut rng, &options),
    };

    return_sample(s)
}

#[savvy]
fn rust_doubly_balanced(
    r_prob: RealSexp,
    r_data: RealSexp,
    r_bal_data: RealSexp,
    r_eps: f64,
    r_bucket_size: i32,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);

    let bucket_size = i32_to_nonzerousize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?
        .set_spreading(aux)?;

    let s = match r_method {
        "local_cube" | &_ => local_cube(&mut rng, &options),
    };

    return_sample(s)
}

#[savvy]
fn rust_spatially_balanced_hierarchical(
    r_prob: RealSexp,
    r_data: RealSexp,
    r_sizes: IntegerSexp,
    r_eps: f64,
    r_bucket_size: i32,
    r_method: &str,
) -> savvy::Result<Sexp> {
    // Returns a matrix with sample indices (0) and groups (1)
    let mut rng = RRng::new();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let bucket_size = i32_to_nonzerousize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_spreading(aux)?;

    let sizes: Vec<usize> = r_sizes
        .iter()
        .map(|&x| i32_to_usize(x))
        .collect::<savvy::Result<_>>()?;

    let s = match r_method {
        "lpm_2" | &_ => hierarchical_lpm_2(&mut rng, &options, &sizes)?,
    };

    let n = sizes.iter().sum();
    let mut return_matrix = OwnedIntegerSexp::new(n * 2)?;

    let mut idx: usize = 0;
    for (i, vec) in s.iter().enumerate() {
        for &j in vec.iter() {
            return_matrix[idx] = usize_to_i32(j)? + 1;
            return_matrix[idx + n] = usize_to_i32(i)?;
            idx += 1;
        }
    }

    return_matrix.set_dim(&[n, 2])?;
    return_matrix.into()
}

#[savvy]
fn rust_balanced_stratified(
    r_prob: RealSexp,
    r_bal_data: RealSexp,
    r_strata: IntegerSexp,
    r_eps: f64,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_balancing(&bal_data)?;

    let s = match r_method {
        "cube" | &_ => cube_stratified(&mut rng, &options, &strata)?,
    };

    return_sample(s)
}

#[savvy]
fn rust_doubly_balanced_stratified(
    r_prob: RealSexp,
    r_data: RealSexp,
    r_bal_data: RealSexp,
    r_strata: IntegerSexp,
    r_eps: f64,
    r_bucket_size: i32,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let bucket_size = i32_to_nonzerousize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(r_prob.as_slice())?
        .set_eps(r_eps)?
        .set_balancing(&bal_data)?
        .set_spreading(aux)?;

    let s = match r_method {
        "local_cube" | &_ => local_cube_stratified(&mut rng, &options, &strata)?,
    };

    return_sample(s)
}

#[savvy]
fn rust_local_mean_variance(
    r_values: RealSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
    r_neighbours: i32,
) -> savvy::Result<Sexp> {
    if r_neighbours <= 0 {
        return f64::NAN.try_into();
    }

    let neighbours = i32_to_nonzerousize(r_neighbours)?;
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(r_prob.as_slice())?.set_spreading(aux)?;

    local_mean_variance(r_values.as_slice(), &options, neighbours)
        .ok_or_else(|| savvy_err!("could not calculate local mean variance"))?
        .try_into()
}

#[savvy]
fn rust_spatial_balance_measure(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| i32_to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(r_prob.as_slice())?.set_spreading(aux)?;

    let v = match r_method {
        "local" => sb_local(&sample, &options, true),
        "local2" => sb_local(&sample, &options, false),
        "energy-distance" => sb_energy(&sample, &options),
        "voronoi" | &_ => sb_voronoi(&sample, &options),
    };

    v.ok_or_else(|| savvy_err!("could not calculate balance measure...invalid sample?"))?
        .try_into()
}

#[savvy]
fn rust_balance_deviation(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| i32_to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(r_prob.as_slice())?.set_spreading(aux)?;

    balance_deviation_spreading(&sample, &options)
        .ok_or_else(|| savvy_err!("no result returned...invalid sample?"))?
        .try_into()
}

#[savvy]
fn rust_pips_from_values(r_values: RealSexp, r_sample_size: i32) -> savvy::Result<Sexp> {
    pips_from_slice(r_values.as_slice(), i32_to_usize(r_sample_size)?)?
        .data()
        .try_into()
}
