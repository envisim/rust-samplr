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

//! Simple random sampling

use envisim_utils::probabilities::ProbabilitiesEqual;
use envisim_utils::random::RandomNumberGenerator;
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};

/// Draw a simple random sample without replacement
///
/// # Examples
/// ```
/// use envisim_samplr::srs::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let opts = SamplingOptions::new_equal(10, 5)?;
/// let s = srs(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
#[inline]
pub fn srs<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesEqual, S, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let population_size = options.population_size();
    let sample_size = options.sample_size();

    if sample_size == 0 {
        return vec![];
    } else if sample_size == population_size {
        return (0usize..population_size).collect();
    }

    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for i in 0..population_size {
        if rng.rusize_to(population_size - i) < sample_size - sample.len() {
            sample.push(i);
        }
    }

    sample
}

/// Draw a simple random sample with replacement
///
/// # Examples
/// ```
/// use envisim_samplr::srs::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let opts = SamplingOptions::new_equal(10, 5)?;
/// let s = srs_with_replacement(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
#[inline]
pub fn srs_with_replacement<R, S, B>(
    rng: &mut R,
    options: &SamplingOptions<'_, ProbabilitiesEqual, S, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let population_size = options.population_size();
    let sample_size = options.sample_size();

    if sample_size == 0 {
        return vec![];
    } else if sample_size == population_size {
        return (0usize..population_size).collect();
    }

    let mut sample: Vec<usize> = (0..sample_size)
        .map(|_| rng.rusize_to(population_size))
        .collect();

    sample.sort_unstable();
    sample
}
