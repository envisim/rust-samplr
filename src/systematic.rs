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

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::Probabilities;

pub use crate::{SampleOptions, SamplingError};

/// Draw a systematic sample, using the provided order
///
/// # Examples
/// ```
/// use envisim_samplr::systematic::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = SampleOptions::new(&p)?.sample(&mut rng, sample)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
#[inline]
pub fn sample<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    options.check_base()?;
    let probabilities = options.probabilities();
    let order: Vec<usize> = (0usize..probabilities.len()).collect();
    from_order(rng.rf64(), probabilities, &order)
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
/// let s = SampleOptions::new(&p)?.sample(&mut rng, sample_random_order)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
#[inline]
pub fn sample_random_order<R>(
    rng: &mut R,
    options: &SampleOptions,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    options.check_base()?;
    let probabilities = options.probabilities();
    let order = shuffle(rng, probabilities.len());
    from_order(rng.rf64(), probabilities, &order)
}

#[inline]
fn from_order(
    rv: f64,
    probabilities: &[f64],
    order: &[usize],
) -> Result<Vec<usize>, SamplingError> {
    Probabilities::check(probabilities)?;

    let mut sample = Vec::<usize>::with_capacity(
        probabilities.iter().fold(0.0, |acc, p| acc + p).ceil() as usize,
    );
    let mut r = rv;
    let mut psum: f64 = 0.0;

    for &id in order.iter() {
        if psum <= r && r <= psum + probabilities[id] {
            sample.push(id);
            r += 1.0;
        }

        psum += probabilities[id];
    }

    Ok(sample)
}

#[inline]
fn shuffle<R>(rng: &mut R, len: usize) -> Vec<usize>
where
    R: RandomNumberGenerator + ?Sized,
{
    let mut order: Vec<usize> = (0..len).collect();

    for i in (1..len).rev() {
        order.swap(i, rng.rusize_to(i + 1));
    }

    order
}
