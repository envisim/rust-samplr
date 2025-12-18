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

//! Unequal probability sampling designs

use envisim_utils::indices::Indices;
use envisim_utils::pips::{
    Probabilities,
    ProbabilitiesUnequal,
};
use envisim_utils::random::RandomNumberGenerator;
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};
use envisim_utils::utils::{
    f64_to_usize,
    usize_to_f64,
};

pub use crate::error::SamplingError;

// Assumes probabilites sum to 1.0
fn draw<R>(rng: &mut R, probabilities: &[f64]) -> usize
where
    R: RandomNumberGenerator,
{
    let population_size = probabilities.len();
    let rv = rng.rf64();
    let mut psum: f64 = 0.0;

    for (i, &p) in probabilities.iter().enumerate() {
        psum += p;

        if rv <= psum {
            return i;
        }
    }

    population_size - 1
}

/// Draw a with replacment sample according to draw probabilities
/// Probabilities must sum to 1.0.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.1; 10];
/// let options = SamplingOptions::new(&p)?;
/// let s = with_replacement(&mut rng, &options, 5)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn with_replacement<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesUnequal, S, B>,
    n: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
{
    let probabilities = options.probabilities().slice();
    let psum: f64 = probabilities.iter().sum();

    if (psum - 1.0).abs() > options.eps() {
        return Err(SamplingError::IncorrectDrawProbabilities);
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut rvs = Vec::<f64>::with_capacity(n);

    for _ in 0..n {
        rvs.push(rng.rf64());
    }

    rvs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sample = Vec::<usize>::with_capacity(n);
    let mut psum: f64 = 0.0;
    let mut rv_iter = rvs.iter();
    let mut rv = *rv_iter.next().unwrap();

    // Add units for which rv is in [psum, psum+p)
    // Go up one p when psum+p < rv
    // Go up one rv when sample has been pushed
    'outer: for (id, &p) in probabilities.iter().enumerate() {
        loop {
            if psum + p <= rv {
                psum += p;
                break;
            }

            if rv < psum + p {
                sample.push(id);

                match rv_iter.next() {
                    Some(v) => {
                        rv = *v;
                        continue;
                    }
                    _ => break 'outer,
                }
            }
        }
    }

    Ok(sample)
}

/// Draw a sample using a sampford design.
/// Probabilities must sum to an integer.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let options = SamplingOptions::new(&p)?;
/// let s = sampford(&mut rng, &options)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn sampford<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesUnequal, S, B>,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
{
    let probabilities = options.probabilities().slice();
    let eps = options.eps();
    let psum: f64 = probabilities.iter().sum();
    let sample_size: usize = if (psum - psum.round()).abs() <= eps {
        f64_to_usize(psum)
    } else {
        return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
    };

    if sample_size == 0 {
        return Ok(vec![]);
    } else if sample_size == 1 {
        return Ok(vec![draw(rng, probabilities.as_ref())]);
    }

    let norm_probs: Vec<f64> = probabilities.iter().map(|&p| p / psum).collect();

    for _ in 0..options.max_iterations().get() {
        let mut sample = poisson_internal(rng, probabilities.as_ref());

        if sample.len() != sample_size - 1 {
            continue;
        }

        let a_unit = draw(rng, &norm_probs);

        // Since sample is ordered, we don't need to check units with
        // higher id than a_unit
        if sample
            .iter()
            .find(|&&id| id >= a_unit)
            .is_some_and(|&id| id != a_unit)
        {
            sample.push(a_unit);
            sample.sort_unstable();
            return Ok(sample);
        }
    }

    Err(SamplingError::MaxIterations(options.max_iterations()))
}

/// Draw a sample using a pareto design.
/// Probabilities must sum to an integer.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let options = SamplingOptions::new(&p)?;
/// let s = pareto(&mut rng, &options)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Rosén, B. (2000).
/// A user’s guide to Pareto pi-ps sampling. R & D Report 2000:6.
/// Stockholm: Statistiska Centralbyrån.
pub fn pareto<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesUnequal, S, B>,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
{
    let probabilities = options.probabilities().slice();
    let eps = options.eps();
    let psum: f64 = probabilities.iter().sum();
    let sample_size: usize = if (psum - psum.round()).abs() <= eps {
        f64_to_usize(psum)
    } else {
        return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
    };

    let q_values: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            let u = rng.rf64();

            if 1.0 - eps < u || p < eps {
                return f64::INFINITY;
            }

            let res = (u * (1.0 - p)) / (p * (1.0 - u));

            if res.is_nan() {
                return f64::INFINITY;
            }

            res
        })
        .collect();

    let mut sample: Vec<usize> = (0..probabilities.len()).collect();
    sample.sort_by(|&a, &b| q_values[a].partial_cmp(&q_values[b]).unwrap());
    sample.truncate(sample_size);
    Ok(sample)
}

/// Draw a sample using a brewer design.
/// Probabilities must sum to an integer.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let opts = SamplingOptions::new(&p)?;
/// let s = brewer(&mut rng, &opts)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn brewer<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesUnequal, S, B>,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
{
    let probabilities = options.probabilities().slice();
    let eps = options.eps();
    let mut psum: f64 = probabilities.iter().sum();
    let mut sample_size: usize = if (psum - psum.round()).abs() <= eps {
        f64_to_usize(psum)
    } else {
        return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
    };
    let mut n_d = psum;
    let mut indices = Indices::with_fill(probabilities.len());
    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for (id, &p) in probabilities.iter().enumerate() {
        if p <= eps {
            indices.remove(id).unwrap();
        } else if 1.0 - eps <= p {
            indices.remove(id).unwrap();
            sample.push(id);
            n_d -= 1.0;
            sample_size -= 1;
        }
    }

    let mut q_probs: Vec<f64> = vec![0.0; probabilities.len()];

    for i in 0..sample_size {
        psum = 0.0;
        for &id in indices.list() {
            let p = probabilities[id];
            q_probs[id] = p * (n_d - p) / (n_d - p * usize_to_f64(sample_size - i + 1));
            psum += q_probs[id];
        }

        for &id in indices.list() {
            q_probs[id] /= psum;
        }

        let a_unit = draw(rng, &q_probs);
        indices.remove(a_unit).unwrap();
        sample.push(a_unit);
        q_probs[a_unit] = 0.0;
        n_d -= probabilities[a_unit];
    }

    sample.sort_unstable();
    Ok(sample)
}

pub(crate) fn poisson_internal<R>(rng: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| rng.rbern(p).and_then(|b| b.then_some(i)))
        .collect()
}

/// Draw a sample using a poisson design.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let opts = SamplingOptions::new(&p)?;
/// let s = poisson(&mut rng, &opts);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn poisson<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
{
    let probabilities = options.probabilities().slice();
    poisson_internal(rng, probabilities.as_ref())
}

/// Draw a sample using a conditional poisson design.
/// Redraws a poisson sample until the fixed sample size is achieved.
/// May terminate after `max_iterations`.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let options = SamplingOptions::new(&p)?;
/// let s = conditional_poisson(&mut rng, &options, 5);
/// # Ok::<(), SamplingError>(())
/// ```
pub fn conditional_poisson<R, P, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, P, S, B>,
    sample_size: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
    P: Probabilities,
{
    let probabilities = options.probabilities().slice();
    let population_size = options.population_size();
    if sample_size > population_size {
        return Err(SamplingError::Options(
            SamplingOptionsError::InvalidSampleSize,
        ));
    } else if sample_size == 0 {
        return Ok(vec![]);
    } else if sample_size == population_size {
        return Ok((0..population_size).collect::<Vec<usize>>());
    }

    for _ in 0..options.max_iterations().get() {
        let s = poisson_internal(rng, probabilities.as_ref());

        if s.len() == sample_size {
            return Ok(s);
        }
    }

    Err(SamplingError::MaxIterations(options.max_iterations()))
}
