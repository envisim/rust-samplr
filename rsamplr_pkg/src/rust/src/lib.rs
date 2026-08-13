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
use envisim_estimate::spatial_balance::SpatialBalance;
use envisim_samplr::cube_method::{
    cube_stratified,
    local_cube_stratified,
};
use envisim_samplr::dbd::DistributionalDesignEvaluators;
use envisim_samplr::pivotal_method::hierarchical_lpm_2;
use envisim_samplr::*;
use envisim_utils::pips::pips_from_slice;
use envisim_utils::sampling_options::SamplingOptions;
use savvy::{
    IntegerSexp,
    OwnedIntegerSexp,
    RealSexp,
    Sexp,
    savvy,
};

mod random;
mod sexp;
mod utils;

use random::*;
use sexp::*;
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
    let options = RealSexpFatPtr::to_sampling_options(r_prob, r_eps, r_max_iter)?;

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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, r_bucket_size)?;

    let options = RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?.set_spreading(aux)?;

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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let sample_size = to_usize(r_sample_size)?;
    let iter = to_nzusize(r_iter)?;

    let dbs_options = DistributionalDesignOptions::new(r_temp)?
        .set_annealing_cooling_rate(r_cooling)?
        .set_spatial_initialization(r_spatial_init);

    let options = SamplingOptions::with_spreading(aux, sample_size)?.set_max_iterations(iter)?;

    match r_method {
        "dbd_tc" => return_sample(options.dbd_tc(&mut rng, dbs_options)?.into_buckets().data()),
        "dbd_circular" | &_ => return_sample(
            options
                .dbd_circular(&mut rng, dbs_options)?
                .into_sequence()
                .as_ref(),
        ),
    }
}

#[savvy]
fn rust_balanced(
    r_prob: RealSexp,
    r_bal_data: RealSexp,
    r_eps: f64,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let mut rng = RRng::new();
    let bal_data = RealSexpFatPtr::to_matrix(r_bal_data)?;
    let options =
        RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?.set_balancing(bal_data)?;

    let s = match r_method {
        "sequential_cube" => options.sequential_cube(&mut rng),
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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, r_bucket_size)?;
    let bal_data = RealSexpFatPtr::to_matrix(r_bal_data)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?
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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, r_bucket_size)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?.set_spreading(aux)?;

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
    let bal_data = RealSexpFatPtr::to_matrix(r_bal_data)?;
    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();
    let options =
        RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?.set_balancing(bal_data)?;

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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, r_bucket_size)?;
    let bal_data = RealSexpFatPtr::to_matrix(r_bal_data)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, r_eps, None)?
        .set_balancing(bal_data)?
        .set_spreading(aux)?;

    let strata: Vec<i64> = r_strata.iter().map(|&x| i64::from(x)).collect();

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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, None, None)?.set_spreading(aux)?;

    let Ok(neighbours) = to_nzusize(r_neighbours) else {
        return f64::NAN.try_into();
    };

    local_mean_variance(r_values.as_slice(), &options, neighbours)?.try_into()
}

#[savvy]
fn rust_spatial_balance_measure(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
    r_method: &str,
) -> savvy::Result<Sexp> {
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, None, None)?.set_spreading(aux)?;

    let sample = to_sample(&r_sample)?;

    let v = match r_method {
        "local" => options.local(&sample, true)?,
        "local2" => options.local(&sample, false)?,
        "energy-distance" => options.energy_distance(&sample),
        "voronoi" | &_ => options.voronoi(&sample)?,
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
    let sample_size = to_usize(r_sample_size)?;
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = SamplingOptions::with_spreading(aux, sample_size)?;

    let sample = to_sample(&r_sample)?;

    let v = match r_method {
        "local" => options.local(&sample, true)?,
        "local2" => options.local(&sample, false)?,
        "energy-distance" => options.energy_distance(&sample),
        "voronoi" | &_ => options.voronoi(&sample)?,
    };

    v.try_into()
}

#[savvy]
fn rust_balance_deviation(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, None, None)?.set_spreading(aux)?;

    let sample = to_sample(&r_sample)?;

    balance_deviation_spreading(&sample, &options)?.try_into()
}

#[savvy]
fn rust_pips_from_values(r_values: RealSexp, r_sample_size: i32) -> savvy::Result<Sexp> {
    let pips: Vec<f64> = pips_from_slice(r_values.as_slice(), to_usize(r_sample_size)?)?.to_raw();
    pips.try_into()
}

#[savvy]
fn rust_spatial_balance_measure_all(
    r_sample: IntegerSexp,
    r_prob: RealSexp,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = RealSexpFatPtr::to_sampling_options(r_prob, None, None)?.set_spreading(aux)?;

    let sample = to_sample(&r_sample)?;

    let bms = vec![
        options.voronoi(&sample)?,
        options.local(&sample, true)?,
        options.local(&sample, false)?,
        options.energy_distance(&sample),
    ];

    bms.try_into()
}

#[savvy]
fn rust_spatial_balance_measure_all_equal(
    r_sample: IntegerSexp,
    r_sample_size: i32,
    r_data: RealSexp,
) -> savvy::Result<Sexp> {
    let sample_size = to_usize(r_sample_size)?;
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let options = SamplingOptions::with_spreading(aux, sample_size)?;

    let sample = to_sample(&r_sample)?;

    let bms = vec![
        options.voronoi(&sample)?,
        options.local(&sample, true)?,
        options.local(&sample, false)?,
        options.energy_distance(&sample),
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
    let aux = RealSexpFatPtr::to_spreading_options(r_data, None)?;
    let sample_size = to_usize(r_sample_size)?;

    let iter_to = to_nzusize(r_iter_to)?;
    let iter_by = to_nzusize(r_iter_by)?;

    let dbs_options = DistributionalDesignOptions::new(r_temp)?
        .set_annealing_cooling_rate(r_cooling)?
        .set_spatial_initialization(r_spatial_init);

    let options = SamplingOptions::with_spreading(aux, sample_size)?;

    let s = match r_method {
        "dbd_tc" => options.dbd_tc_evaluator(&mut rng, dbs_options, iter_to, iter_by),
        "dbd_circular" | &_ => {
            options.dbd_circular_evaluator(&mut rng, dbs_options, iter_to, iter_by)
        }
    }?;

    s.try_into()
}
