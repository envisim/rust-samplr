// Copyright (C) 2026 Wilmer Prentius.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Savvy-R-wrappers for [`envisim_samplr`] and [`envisim_estimate`]

#![expect(clippy::wildcard_imports, reason = "need everything")]
#![expect(clippy::too_many_arguments, reason = "can only send Sexp")]

use envisim_estimate::balance::balance_deviation_spreading;
use envisim_estimate::horvitz_thompson::local_mean_variance;
use envisim_estimate::spatial_balance::{
    energy_distance as sb_energy,
    local as sb_local,
    voronoi as sb_voronoi,
};
use envisim_samplr::cube_method::{
    cube_stratified,
    local_cube_stratified,
};
use envisim_samplr::dbd::DistributionalDesignEvaluators;
use envisim_samplr::pivotal_method::hierarchical_lpm_2;
use envisim_samplr::*;
use envisim_utils::pips::pips_from_slice;
use envisim_utils::probabilities::ProbabilityStore;
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
    r_sample_size: i32,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let prob = r_prob.as_slice();
    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_max_iterations(to_nzusize(r_max_iter)?)?;

    let s = match r_method {
        "spm" => options.spm(&mut rng),
        "cps" => options.cps(&mut rng),
        "poisson" => options.poisson(&mut rng),
        "conditional_poisson" => {
            let sample_size = to_usize(r_sample_size)?;
            options.conditional_poisson(&mut rng, sample_size)?
        }
        "systematic" => options.systematic(&mut rng),
        "systematic_random_order" => options.systematic_random_order(&mut rng),
        "brewer" => options.brewer(&mut rng)?,
        "pareto" => options.pareto(&mut rng)?,
        "sampford" => options.sampford(&mut rng)?,
        "rpm" | &_ => options.rpm(&mut rng),
    };

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
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let bucket_size = to_usize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_spreading(aux)?;

    let s = match r_method {
        "lpm_1" => options.lpm_1(&mut rng),
        "lpm_1s" => options.lpm_1s(&mut rng),
        "scps" => options.scps(&mut rng),
        "lcps" => options.lcps(&mut rng),
        "lpm_2" | &_ => options.lpm_2(&mut rng),
    };

    return_sample(s)
}

#[savvy]
fn rust_distributionally_balanced_design(
    r_sample_size: i32,
    r_data: RealSexp,
    r_temp: f64,
    r_cooling: f64,
    r_iter: i32,
    r_spatial_init: bool,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let population_size = get_nrow(&r_data)?;
    let sample_size = to_usize(r_sample_size)?;
    let data = to_matrix(r_data.as_slice(), population_size);
    let iter = to_nzusize(r_iter)?;

    let dbs_options = DistributionalDesignOptions::new(r_temp)?
        .set_annealing_cooling_rate(r_cooling)?
        .set_spatial_initialization(r_spatial_init);

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new_equal(population_size, sample_size)?
        .set_spreading(aux)?
        .set_max_iterations(iter)?;

    let s = match r_method {
        "dbd_tc" => options.dbd_tc(&mut rng, dbs_options)?.into_buckets(),
        "dbd_circular" | &_ => options.dbd_circular(&mut rng, dbs_options)?.into_sequence(),
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
    let prob = r_prob.as_slice();
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);

    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?;

    let s = match r_method {
        "cube" | &_ => options.cube(&mut rng),
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
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);

    let bucket_size = to_usize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?
        .set_spreading(aux)?;

    let s = match r_method {
        "local_cube" | &_ => options.local_cube(&mut rng),
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
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let bucket_size = to_usize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_spreading(aux)?;

    let sizes: Vec<usize> = r_sizes
        .iter()
        .map(|&x| to_usize(x))
        .collect::<savvy::Result<_>>()?;

    let s = match r_method {
        "lpm_2" | &_ => hierarchical_lpm_2(&mut rng, &options, &sizes)?,
    };

    let n = sizes.iter().sum();
    let mut return_matrix = OwnedIntegerSexp::new(n * 2)?;

    let mut idx: usize = 0;
    for (i, vec) in s.iter().enumerate() {
        for &j in vec {
            return_matrix[idx] = to_i32(j)? + 1;
            return_matrix[idx + n] = to_i32(i)?;
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
    let prob = r_prob.as_slice();
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?;

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
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let bal_data = to_matrix(r_bal_data.as_slice(), get_nrow(&r_bal_data)?);
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

    let bucket_size = to_usize(r_bucket_size)?;
    let aux = SpreadingOptions::new(data).set_bucket_size(bucket_size)?;
    let options = SamplingOptions::new(prob.into())?
        .set_eps(r_eps)?
        .set_balancing(bal_data)?
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

    let neighbours = to_nzusize(r_neighbours)?;
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(prob.into())?.set_spreading(aux)?;

    local_mean_variance(r_values.as_slice(), &options, neighbours)?.try_into()
}

#[savvy]
fn rust_spatial_balance_measure(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(prob.into())?.set_spreading(aux)?;

    let v = match r_method {
        "local" => sb_local(&sample, &options, true)?,
        "local2" => sb_local(&sample, &options, false)?,
        "energy-distance" => sb_energy(&sample, &options),
        "voronoi" | &_ => sb_voronoi(&sample, &options)?,
    };

    v.try_into()
}

#[savvy]
fn rust_spatial_balance_measure_equal(
    r_sample: IntegerSexp,
    r_sample_size: i32,
    r_data: RealSexp,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;
    let sample_size = to_usize(r_sample_size)?;
    let population_size = data.nrow();

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new_equal(population_size, sample_size)?.set_spreading(aux)?;

    let v = match r_method {
        "local" => sb_local(&sample, &options, true)?,
        "local2" => sb_local(&sample, &options, false)?,
        "energy-distance" => sb_energy(&sample, &options),
        "voronoi" | &_ => sb_voronoi(&sample, &options)?,
    };

    v.try_into()
}

#[savvy]
fn rust_balance_deviation(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(prob.into())?.set_spreading(aux)?;

    balance_deviation_spreading(&sample, &options)?.try_into()
}

#[savvy]
fn rust_pips_from_values(r_values: RealSexp, r_sample_size: i32) -> savvy::Result<Sexp> {
    pips_from_slice(r_values.as_slice(), to_usize(r_sample_size)?)?
        .data()
        .try_into()
}

#[savvy]
fn rust_spatial_balance_measure_all(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let prob = r_prob.as_slice();
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new(prob.into())?.set_spreading(aux)?;

    let bms = vec![
        sb_voronoi(&sample, &options)?,
        sb_local(&sample, &options, true)?,
        sb_local(&sample, &options, false)?,
        sb_energy(&sample, &options),
    ];

    bms.try_into()
}

#[savvy]
fn rust_spatial_balance_measure_all_equal(
    r_sample: IntegerSexp,
    r_sample_size: i32,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let data = to_matrix(r_data.as_slice(), get_nrow(&r_data)?);
    let sample: Vec<usize> = r_sample
        .iter()
        .map(|&x| to_usize(x - 1))
        .collect::<savvy::Result<_>>()?;
    let sample_size = to_usize(r_sample_size)?;
    let population_size = data.nrow();

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new_equal(population_size, sample_size)?.set_spreading(aux)?;

    let bms = vec![
        sb_voronoi(&sample, &options)?,
        sb_local(&sample, &options, true)?,
        sb_local(&sample, &options, false)?,
        sb_energy(&sample, &options),
    ];

    bms.try_into()
}

#[savvy]
fn rust_distributionally_balanced_design_iter(
    r_sample_size: i32,
    r_data: RealSexp,
    r_temp: f64,
    r_cooling: f64,
    r_spatial_init: bool,
    r_iter_to: i32,
    r_iter_by: i32,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let population_size = get_nrow(&r_data)?;
    let sample_size = to_usize(r_sample_size)?;
    let data = to_matrix(r_data.as_slice(), population_size);
    let iter_to = to_nzusize(r_iter_to)?;
    let iter_by = to_nzusize(r_iter_by)?;

    let dbs_options = DistributionalDesignOptions::new(r_temp)?
        .set_annealing_cooling_rate(r_cooling)?
        .set_spatial_initialization(r_spatial_init);

    let aux = SpreadingOptions::new(data);
    let options = SamplingOptions::new_equal(population_size, sample_size)?.set_spreading(aux)?;

    let s = match r_method {
        "dbd_tc" => options.dbd_tc_evaluator(&mut rng, dbs_options, iter_to, iter_by),
        "dbd_circular" | &_ => {
            options.dbd_circular_evaluator(&mut rng, dbs_options, iter_to, iter_by)
        }
    }?;

    s.try_into()
}
