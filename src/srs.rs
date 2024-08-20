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

//! Simple random sampling

use envisim_utils::error::{InputError, SamplingError};
use rand::Rng;

/// Draw a simple random sample without replacement
///
/// # Examples
/// ```
/// use envisim_samplr::srs::sample;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// assert_eq!(sample(&mut rng, 5, 10).unwrap().len(), 5);
/// ```
#[inline]
pub fn sample<R>(
    rng: &mut R,
    sample_size: usize,
    population_size: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    InputError::check_sample_size(sample_size, population_size)?;

    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for i in 0..population_size {
        if rng.gen_range(0..(population_size - i)) < sample_size - sample.len() {
            sample.push(i);
        }
    }

    Ok(sample)
}

/// Draw a simple random sample with replacement
///
/// # Examples
/// ```
/// use envisim_samplr::srs::sample_with_replacement;
/// use rand::{rngs::SmallRng, SeedableRng};
/// let mut rng = SmallRng::from_entropy();
/// assert_eq!(sample_with_replacement(&mut rng, 5, 10).unwrap().len(), 5);
/// ```
#[inline]
pub fn sample_with_replacement<R>(
    rng: &mut R,
    sample_size: usize,
    population_size: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    InputError::check_sample_size(sample_size, population_size)?;

    let mut sample: Vec<usize> = (0..sample_size)
        .map(|_| rng.gen_range(0..population_size))
        .collect();

    sample.sort_unstable();
    Ok(sample)
}
