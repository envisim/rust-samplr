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

use envisim_utils::Number;
use envisim_utils::random::{
    Rand,
    Rng,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    EqualProbabilityOptions,
    ProbabilityOptions,
    SamplingOptionsRng,
    UnequalProbabilityOptions,
};

pub use crate::error::SamplingError;
use crate::utils::shuffled_indices;

/// Draws a systematic sample from the provided order using equal probabilities
#[must_use]
#[inline]
fn from_order_equal<R>(
    rng: &mut R,
    options: &EqualProbabilityOptions,
    order: &[usize],
) -> Vec<usize>
where
    R: Rand<usize>,
{
    let mut sample = Vec::<usize>::with_capacity(options.sample_size() + 1);
    let mut r = rng.rand_to(options.population_size().get());
    let mut psum: usize = 0;

    for &id in order {
        let pnext = psum + options.sample_size();
        if psum <= r && r < pnext {
            sample.push(id);
            r += options.population_size().get();
        }
        psum = pnext;
    }

    sample
}

/// Draws a systematic sample from the provided order using unequal probabilities
#[must_use]
#[inline]
fn from_order<'bprob, R, N>(
    rng: &mut R,
    options: &UnequalProbabilityOptions<'bprob, N>,
    order: &[usize],
) -> Vec<usize>
where
    R: Rand<N>,
    N: Number,
    UnequalProbabilityOptions<'bprob, N>: ProbabilityOptions<Native = N>,
{
    let (probs, pmax) = options.to_slice();
    let mut sample = Vec::<usize>::with_capacity(options.sample_size());
    let mut r = rng.rand_to(pmax);
    let mut psum: N = N::ZERO;

    for &id in order {
        let pnext = psum + probs[id];
        if psum <= r && r < pnext {
            sample.push(id);
            r += pmax;
        }

        psum = pnext;
    }

    sample
}

pub trait SystematicSampling<R>
where
    R: Rng,
{
    /// Draw a systematic sample, using the provided order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a systematic sample, using a random order
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.systematic_random_order(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic_random_order(&self, rng: &mut R) -> Vec<usize>;
}
impl<R, AUX, BAL> SystematicSampling<R> for SamplingOptions<EqualProbabilityOptions, AUX, BAL>
where
    R: SamplingOptionsRng<EqualProbabilityOptions>,
{
    #[inline]
    fn systematic(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let order: Vec<usize> = (0..population_size.get()).collect();
        from_order_equal(rng, self.probabilities(), &order)
    }
    #[inline]
    fn systematic_random_order(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let order = shuffled_indices(rng, population_size);
        from_order_equal(rng, self.probabilities(), &order)
    }
}
impl<'bprob, R, PROB, AUX, BAL> SystematicSampling<R>
    for SamplingOptions<UnequalProbabilityOptions<'bprob, PROB>, AUX, BAL>
where
    R: SamplingOptionsRng<UnequalProbabilityOptions<'bprob, PROB>>,
    UnequalProbabilityOptions<'bprob, PROB>: ProbabilityOptions<Native = PROB>,
    PROB: Number,
{
    #[inline]
    fn systematic(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let order: Vec<usize> = (0..population_size.get()).collect();
        from_order(rng, self.probabilities(), &order)
    }
    #[inline]
    fn systematic_random_order(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let order = shuffled_indices(rng, population_size);
        from_order(rng, self.probabilities(), &order)
    }
}
