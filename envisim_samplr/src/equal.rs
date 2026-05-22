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
//!
//! Implements [`EqualProbabilitySampling`] for [`SamplingOptions`].

use envisim_utils::random::{
    FloatRng,
    Rand,
    RandSlice,
};
use envisim_utils::sampling_options::ProbabilitySpecEqual;
pub use envisim_utils::sampling_options::SamplingOptions;

pub use crate::error::SamplingError;

pub trait EqualProbabilitySampling {
    /// Draw a simple random sample without replacement
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let s = SamplingOptions::new_equal(10, 5)?.srs(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn srs<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: FloatRng;
    /// Draw a simple random sample with replacement
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let s = SamplingOptions::new_equal(10, 5)?.srs_with_replacement(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn srs_with_replacement<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: FloatRng;
    /// Draw a sample using Bernoulli sampling
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let s = SamplingOptions::new_equal(10, 5)?.bernoulli(&mut rng);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn bernoulli<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: FloatRng;
}
impl<AUX, BAL> EqualProbabilitySampling for SamplingOptions<ProbabilitySpecEqual, AUX, BAL> {
    #[must_use]
    #[inline]
    fn srs<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: Rand<usize>,
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
            if rng.rand_in(0..(population_size.get() - i)) < sample_size - sample.len() {
                sample.push(i);
            }
        }

        sample
    }
    #[must_use]
    #[inline]
    fn srs_with_replacement<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandSlice<usize>,
    {
        let population_size = self.population_size();
        let sample_size = self.sample_size();

        if sample_size == 0 {
            return vec![];
        } else if sample_size == population_size.get() {
            return (0..population_size.get()).collect();
        }

        let mut sample = vec![0; sample_size];
        rng.rand_to_n(&mut sample, population_size.get());
        sample.sort_unstable();
        sample
    }
    #[must_use]
    #[inline]
    fn bernoulli<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandSlice<usize>,
    {
        let population_size = self.population_size().get();
        let sample_size = self.sample_size();

        let mut rands = vec![0; population_size];
        rng.rand_in_n(&mut rands, 0..population_size);
        rands
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v < sample_size).then_some(i))
            .collect()
    }
}
