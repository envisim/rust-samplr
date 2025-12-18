// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
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

//! Systematic sampling designs

use envisim_utils::pips::Probabilities;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::ProbabilitySpec;
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};
use envisim_utils::utils::f64_to_usize;

pub use crate::error::SamplingError;

/// Draw a systematic sample, using the provided order
///
/// # Examples
/// ```
/// use envisim_samplr::systematic::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let opts = SamplingOptions::new(&p)?;
/// let s = systematic(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn systematic<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
{
    let probabilities_opts = options.probabilities();
    let population_size = options.population_size();
    let order: Vec<usize> = (0..population_size).collect();
    match probabilities_opts.spec() {
        ProbabilitySpec::Equal { sample_size } => {
            from_order_equal(rng, population_size, *sample_size, &order)
        }
        ProbabilitySpec::Unequal { values } => from_order(rng, values.as_ref(), &order),
        _ => panic!("ProbabilitySpec not implemented"),
    }
}

/// Draw a systematic sample, using a random order
///
/// # Examples
/// ```
/// use envisim_samplr::systematic::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let opts = SamplingOptions::new(&p)?;
/// let s = systematic_random_order(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn systematic_random_order<R, P, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, P, S, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
{
    let probabilities_opts = options.probabilities();
    let population_size = options.population_size();
    let order = shuffle(rng, population_size);
    match probabilities_opts.spec() {
        ProbabilitySpec::Equal { sample_size } => {
            from_order_equal(rng, population_size, *sample_size, &order)
        }
        ProbabilitySpec::Unequal { values } => from_order(rng, values.as_ref(), &order),
        _ => panic!("ProbabilitySpec not implemented"),
    }
}

fn from_order_equal<R: RandomNumberGenerator>(
    rng: &mut R,
    population_size: usize,
    sample_size: usize,
    order: &[usize],
) -> Vec<usize> {
    let mut sample = Vec::<usize>::with_capacity(sample_size + 1);
    let mut r = rng.rusize_to(population_size);
    let mut psum: usize = 0;

    for &id in order.iter() {
        let pnext = psum + sample_size;
        if psum <= r && r < pnext {
            sample.push(id);
            r += population_size;
        }
        psum = pnext;
    }

    sample
}

fn from_order<R: RandomNumberGenerator>(
    rng: &mut R,
    probabilities: &[f64],
    order: &[usize],
) -> Vec<usize> {
    let mut sample =
        Vec::<usize>::with_capacity(f64_to_usize(probabilities.iter().sum::<f64>().ceil()));
    let mut r = rng.rf64();
    let mut psum: f64 = 0.0;

    for &id in order.iter() {
        let pnext = psum + probabilities[id];
        if psum <= r && r < pnext {
            sample.push(id);
            r += 1.0;
        }

        psum = pnext;
    }

    sample
}

fn shuffle<R>(rng: &mut R, len: usize) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut order: Vec<usize> = (0..len).collect();

    for i in (1..len).rev() {
        order.swap(i, rng.rusize_to(i + 1));
    }

    order
}
