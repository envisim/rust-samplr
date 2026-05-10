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

//! Systematic sampling designs
//!
//! Implements [`SystematicSampling`] for [`SamplingOptions`].

use envisim_utils::random::RandomNumberGenerator;
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
};
use num_traits::ToPrimitive;

pub use crate::error::SamplingError;
use crate::utils::shuffled_indices;

/// Draws a systematic sample from the provided order using equal probabilities
#[must_use]
#[inline]
fn from_order_equal<R>(rng: &mut R, spec: ProbabilitySpecEqual, order: &[usize]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut sample = Vec::<usize>::with_capacity(spec.sample_size() + 1);
    let mut r = rng.rusize_to(spec.population_size().get());
    let mut psum: usize = 0;

    for &id in order {
        let pnext = psum + spec.sample_size();
        if psum <= r && r < pnext {
            sample.push(id);
            r += spec.population_size().get();
        }
        psum = pnext;
    }

    sample
}

/// Draws a systematic sample from the provided order using unequal probabilities
#[must_use]
#[inline]
fn from_order<R>(rng: &mut R, probabilities: &[f64], order: &[usize]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut sample = Vec::<usize>::with_capacity(
        probabilities
            .iter()
            .sum::<f64>()
            .ceil()
            .to_usize()
            .expect("probability sum to be contained in usize"),
    );
    let mut r = rng.rf64();
    let mut psum: f64 = 0.0;

    for &id in order {
        let pnext = psum + probabilities[id];
        if psum <= r && r < pnext {
            sample.push(id);
            r += 1.0;
        }

        psum = pnext;
    }

    sample
}

pub trait SystematicSampling {
    /// Draw a systematic sample, using the provided order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
    /// Draw a systematic sample, using a random order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic_random_order(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic_random_order<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
}
impl<PS, AUX, BAL> SystematicSampling for SamplingOptions<PS, AUX, BAL>
where
    PS: ProbabilitySpec,
{
    #[inline]
    fn systematic<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        let population_size = self.population_size();
        let order: Vec<usize> = (0..population_size.get()).collect();

        if let Some(spec) = self.probabilities().as_equal() {
            from_order_equal(rng, spec, &order)
        } else {
            from_order(rng, self.probabilities().as_f64_slice().as_ref(), &order)
        }
    }
    #[inline]
    fn systematic_random_order<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        let population_size = self.population_size();
        let order = shuffled_indices(rng, population_size);

        if let Some(spec) = self.probabilities().as_equal() {
            from_order_equal(rng, spec, &order)
        } else {
            from_order(rng, self.probabilities().as_f64_slice().as_ref(), &order)
        }
    }
}
