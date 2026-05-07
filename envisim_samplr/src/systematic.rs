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

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
    SamplingOptions,
};
use num_traits::ToPrimitive;

pub use crate::error::SamplingError;
use crate::utils::shuffled_indices;

fn from_order_equal<R>(rng: &mut R, spec: ProbabilitySpecEqual, order: &[usize]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut sample = Vec::<usize>::with_capacity(spec.sample_size() + 1);
    let mut r = rng.rusize_to(spec.population_size().get());
    let mut psum: usize = 0;

    for &id in order.iter() {
        let pnext = psum + spec.sample_size();
        if psum <= r && r < pnext {
            sample.push(id);
            r += spec.population_size().get();
        }
        psum = pnext;
    }

    sample
}

fn from_order<R>(rng: &mut R, probabilities: &[f64], order: &[usize]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut sample =
        Vec::<usize>::with_capacity(probabilities.iter().sum::<f64>().ceil().to_usize().unwrap());
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

pub trait SystematicSampling {
    fn systematic<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
    fn systematic_random_order<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
}
impl<PS, SOP, BOP> SystematicSampling for SamplingOptions<PS, SOP, BOP>
where
    PS: ProbabilitySpec,
{
    /// Draw a systematic sample, using the provided order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
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
    /// Draw a systematic sample, using a random order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic_random_order(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
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
