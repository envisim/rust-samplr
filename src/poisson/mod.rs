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

//! Poisson method designs

use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::probability::Probabilities;
use rand::Rng;

// Re-export
mod correlated_poisson;
pub use correlated_poisson::*;

#[inline]
pub(crate) fn internal<R>(rng: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: Rng + ?Sized,
{
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| (rng.gen::<f64>() <= p).then_some(i))
        .collect()
}

/// Draw a sample using a poisson design.
///
/// # Examples
/// ```
/// use envisim_samplr::poisson::sample;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// sample(&mut rng, &p).unwrap();
/// ```
pub fn sample<R>(rng: &mut R, probabilities: &[f64]) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    Probabilities::check(probabilities)?;
    Ok(internal(rng, probabilities))
}

/// Draw a sample using a conditional poisson design.
/// Redraws a poisson sample until the fixed sample size is achieved.
/// May terminate after `max_iterations`.
///
/// # Examples
/// ```
/// use envisim_samplr::poisson::conditional;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// let p: [f64; 10] = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// conditional(&mut rng, &p, 5, 1000);
/// ```
pub fn conditional<R>(
    rng: &mut R,
    probabilities: &[f64],
    sample_size: usize,
    max_iterations: u32,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let population_size = probabilities.len();
    Probabilities::check(probabilities)
        .and(InputError::check_sample_size(sample_size, population_size))?;

    let mut iterations: u32 = 0;

    while iterations < max_iterations {
        let s = internal(rng, probabilities);

        if s.len() == sample_size {
            return Ok(s);
        }

        iterations += 1;
    }

    Err(SamplingError::MaxIterations)
}
