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

//! Simple random sampling

use std::iter::repeat_with;

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::ProbabilitySpecEqual;
pub use envisim_utils::sampling_options::SamplingOptions;

pub use crate::error::SamplingError;

pub trait EqualProbabilitySampling {
    fn srs<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
    fn srs_with_replacement<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
    fn bernoulli<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
}
impl<AUX, BAL> EqualProbabilitySampling for SamplingOptions<ProbabilitySpecEqual, AUX, BAL> {
    /// Draw a simple random sample without replacement
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::from_os_rng();
    /// let s = SamplingOptions::new_equal(10, 5)?.srs(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    #[must_use]
    #[inline]
    fn srs<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        let population_size = self.population_size();
        let sample_size = self.sample_size();

        if sample_size == 0 {
            return vec![];
        } else if sample_size == population_size.get() {
            return (0..population_size.get()).collect();
        }

        let mut sample = Vec::<usize>::with_capacity(sample_size);

        for i in 0..population_size.get() {
            if rng.rusize_to(population_size.get() - i) < sample_size - sample.len() {
                sample.push(i);
            }
        }

        sample
    }
    /// Draw a simple random sample with replacement
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::from_os_rng();
    /// let s = SamplingOptions::new_equal(10, 5)?.srs_with_replacement(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    #[must_use]
    #[inline]
    fn srs_with_replacement<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        let population_size = self.population_size();
        let sample_size = self.sample_size();

        if sample_size == 0 {
            return vec![];
        } else if sample_size == population_size.get() {
            return (0..population_size.get()).collect();
        }

        let mut sample: Vec<usize> = repeat_with(|| rng.rusize_to(population_size.get()))
            .take(sample_size)
            .collect();

        sample.sort_unstable();
        sample
    }
    /// Draw a sample using Bernoulli sampling
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::from_os_rng();
    /// let s = SamplingOptions::new_equal(10, 5)?.bernoulli(&mut rng);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    #[must_use]
    #[inline]
    fn bernoulli<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        let population_size = self.population_size().get();
        let sample_size = self.sample_size();

        let mut sample = Vec::with_capacity(population_size);

        for i in 0..population_size {
            if rng.rusize_to(population_size) < sample_size {
                sample.push(i);
            }
        }

        sample
    }
}
