// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

use crate::poisson;
use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::indices::Indices;
use envisim_utils::probability::Probabilities;
use envisim_utils::utils::{sum, usize_to_f64};
use rand::Rng;

// Assumes probabilites sum to 1.0
#[inline]
fn draw<R>(rng: &mut R, probabilities: &[f64]) -> usize
where
    R: Rng + ?Sized,
{
    let population_size = probabilities.len();
    let rv = rng.gen::<f64>();
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
/// use envisim_samplr::unequal::with_replacement;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.1; 10];
/// assert_eq!(with_replacement(&mut rng, &p, 1e-12, 5).unwrap().len(), 5);
/// ```
#[inline]
pub fn with_replacement<R>(
    rng: &mut R,
    probabilities: &[f64],
    eps: f64,
    n: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    Probabilities::check(probabilities)?;
    if !(1.0 - eps..1.0 + eps).contains(&sum(probabilities)) {
        return Err(SamplingError::Input(InputError::NotInteger));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut rvs = Vec::<f64>::with_capacity(n);

    for _ in 0..n {
        rvs.push(rng.gen::<f64>());
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
/// use envisim_samplr::unequal::sampford;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// assert_eq!(sampford(&mut rng, &p, 1e-12, 1000).unwrap().len(), 5);
/// ```
#[inline]
pub fn sampford<R>(
    rng: &mut R,
    probabilities: &[f64],
    eps: f64,
    max_iterations: u32,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;
    let sample_size = psum.round() as usize;

    if sample_size == 0 {
        return Ok(vec![]);
    } else if sample_size == 1 {
        return Ok(vec![draw(rng, probabilities)]);
    }

    let norm_probs: Vec<f64> = probabilities.iter().map(|&p| p / psum).collect();

    for _ in 0..max_iterations {
        let mut sample = poisson::internal(rng, probabilities);

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

    Err(SamplingError::MaxIterations)
}

/// Draw a sample using a pareto design.
/// Probabilities must sum to an integer.
///
/// # Examples
/// ```
/// use envisim_samplr::unequal::pareto;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// assert_eq!(pareto(&mut rng, &p, 1e-12).unwrap().len(), 5);
/// ```
///
/// # References
/// Rosén, B. (2000).
/// A user’s guide to Pareto pi-ps sampling. R & D Report 2000:6.
/// Stockholm: Statistiska Centralbyrån.
#[inline]
pub fn pareto<R>(rng: &mut R, probabilities: &[f64], eps: f64) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;

    let sample_size = psum.round() as usize;

    let q_values: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            let u = rng.gen::<f64>();

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
/// use envisim_samplr::unequal::brewer;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// assert_eq!(brewer(&mut rng, &p, 1e-12).unwrap().len(), 5);
/// ```
#[inline]
pub fn brewer<R>(rng: &mut R, probabilities: &[f64], eps: f64) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let mut psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;

    let mut sample_size = psum.round() as usize;
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
