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

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    ProbabilitySpecEqual,
    SamplingOptions,
};

pub trait EqualProbabilitySampling {
    fn srs<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize>;
    fn srs_with_replacement<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize>;
    fn bernoulli<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize>;
}
impl<'a> EqualProbabilitySampling for SamplingOptions<'a, ProbabilitySpecEqual> {
    /// Draw a simple random sample without replacement
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let opts = SamplingOptions::new_equal(10, 5)?;
    /// let s = opts.srs(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn srs<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let sample_size = self.sample_size();

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
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let opts = SamplingOptions::new_equal(10, 5)?;
    /// let s = opts.srs_with_replacement(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn srs_with_replacement<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
        let sample_size = self.sample_size();

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
    /// Draw a sample using Bernoulli sampling
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let opts = SamplingOptions::new_equal(10, 5)?;
    /// let s = opts.bernoulli(&mut rng);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn bernoulli<R: RandomNumberGenerator>(&self, rng: &mut R) -> Vec<usize> {
        let population_size = self.population_size();
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
