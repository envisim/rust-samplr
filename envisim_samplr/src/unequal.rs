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

//! Unequal probability sampling designs
//!
//! Implements [`UnequalProbabilitySampling`] for [`SamplingOptions`].
//!
//! # References
//! Rosén, B. (2000).
//! A user’s guide to Pareto pi-ps sampling. R & D Report 2000:6.
//! Stockholm: Statistiska Centralbyrån.

use envisim_utils::Number;
use envisim_utils::indices::Indices;
use envisim_utils::random::{
    Rand,
    Rng,
};
use envisim_utils::sample::Sample;
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    ProbabilityOptions,
    SamplingOptionsError,
    SamplingOptionsRng,
    UnequalProbabilityOptions,
};
use num_traits::ToPrimitive;

pub use crate::error::SamplingError;
use crate::error::SamplingResult;
use crate::utils::poisson_internal;

/// Draws a single unit using pps
/// Assumes probabilites sum to 1.0
#[must_use]
#[inline]
fn draw<R>(rng: &mut R, probabilities: &[f64]) -> usize
where
    R: Rand<f64>,
{
    let population_size = probabilities.len();
    let rv = rng.rand();
    let mut psum: f64 = 0.0;

    for (i, &p) in probabilities.iter().enumerate() {
        psum += p;

        if rv <= psum {
            return i;
        }
    }

    population_size - 1
}

pub trait UnequalProbabilitySampling<R>
where
    R: Rng,
{
    /// Draw a with replacment sample according to draw probabilities
    /// Probabilities must sum to 1.0.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.1; 10];
    /// let s = SamplingOptions::new(p.into())?.with_replacement(&mut rng, 5)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to 1.0.
    fn with_replacement(&self, rng: &mut R, n: usize) -> SamplingResult<Vec<usize>>;
    /// Draw a sample using a sampford design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.sampford(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn sampford(&self, rng: &mut R) -> SamplingResult<Vec<usize>>;
    /// Draw a sample using a pareto design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.pareto(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn pareto(&self, rng: &mut R) -> SamplingResult<Vec<usize>>;
    /// Draw a sample using a brewer design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.brewer(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn brewer(&self, rng: &mut R) -> SamplingResult<Vec<usize>>;
    /// Draw a sample using a poisson design.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.poisson(&mut rng);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn poisson(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using a conditional poisson design.
    /// Redraws a poisson sample until the fixed sample size is achieved.
    /// May terminate after `max_iterations`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p.into())?.conditional_poisson(&mut rng, 5)?;
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if `sample_size` is larger than the population size.
    fn conditional_poisson(&self, rng: &mut R, sample_size: usize) -> SamplingResult<Vec<usize>>;
}
impl<'bprobs, R, PROB, AUX, BAL> UnequalProbabilitySampling<R>
    for SamplingOptions<UnequalProbabilityOptions<'bprobs, PROB>, AUX, BAL>
where
    R: SamplingOptionsRng<UnequalProbabilityOptions<'bprobs, PROB>>,
    UnequalProbabilityOptions<'bprobs, PROB>: ProbabilityOptions<Real = f64>,
    PROB: Number,
{
    #[inline]
    fn with_replacement(&self, rng: &mut R, n: usize) -> SamplingResult<Vec<usize>> {
        let probabilities = self.probabilities().to_slice_real();
        if !self
            .eps()
            .difference_is_zero(self.probabilities().sample_size_real(), 1.0)
        {
            return Err(SamplingError::IncorrectDrawProbabilities);
        }

        if n == 0 {
            return Ok(vec![]);
        }

        let mut rvs = Vec::<f64>::with_capacity(n);

        for _ in 0..n {
            rvs.push(rng.rand());
        }

        rvs.sort_unstable_by(|a, b| a.partial_cmp(b).expect("rvs to not be NaN"));

        let mut sample = Vec::<usize>::with_capacity(n);
        let mut psum: f64 = 0.0;
        let mut rv_iter = rvs.iter();
        let mut rv = *rv_iter.next().expect("at least one rv to exist");

        // Add units for which rv is in [psum, psum+p)
        // Go up one p when psum+p < rv
        // Go up one rv when sample has been pushed
        'outer: for (id, &p) in probabilities.iter().enumerate() {
            loop {
                if psum + p <= rv {
                    psum += p;
                    break;
                }

                if rv < psum + p {
                    sample.push(id);

                    match rv_iter.next() {
                        Some(v) => {
                            rv = *v;
                            continue;
                        }
                        _ => break 'outer,
                    }
                }
            }
        }

        Ok(sample)
    }
    #[inline]
    fn sampford(&self, rng: &mut R) -> SamplingResult<Vec<usize>> {
        let probabilities = self.probabilities().to_slice_real();
        let psum = self.probabilities().sample_size_real();
        if !self.eps().difference_is_zero(psum, psum.round()) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }
        let sample_size = psum
            .to_usize()
            .expect("probability sum to convert to usize");

        if sample_size == 0 {
            return Ok(vec![]);
        } else if sample_size == 1 {
            return Ok(vec![draw(rng, probabilities.as_ref())]);
        }

        let norm_probs: Vec<f64> = probabilities.iter().map(|&p| p / psum).collect();

        for _ in 0..self.max_iterations().get() {
            let mut sample = poisson_internal(rng, probabilities.as_ref());

            if sample.len() != sample_size - 1 {
                continue;
            }

            let a_unit = draw(rng, &norm_probs);

            // Since sample is ordered, we don't need to check units with
            // higher id than a_unit
            if let Err(pos) = sample.binary_search(&a_unit) {
                sample.insert(pos, a_unit);
                return Ok(sample);
            }
        }

        Err(SamplingError::MaxIterations(self.max_iterations()))
    }
    #[inline]
    fn pareto(&self, rng: &mut R) -> SamplingResult<Vec<usize>> {
        let probabilities = self.probabilities().to_slice_real();
        let eps = self.eps();
        let psum = self.probabilities().sample_size_real();
        if !self.eps().difference_is_zero(psum, psum.round()) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }
        let sample_size = psum
            .to_usize()
            .expect("probability sum to convert to usize");

        let q_values: Vec<f64> = probabilities
            .iter()
            .map(|&p| {
                let u: f64 = rng.rand();

                if eps.is_zero(p) || eps.is_zero(1.0 - u) {
                    return f64::INFINITY;
                }

                let res = (u * (1.0 - p)) / (p * (1.0 - u));

                if res.is_nan() {
                    return f64::INFINITY;
                }

                res
            })
            .collect();

        let mut sample: Vec<usize> = (0..probabilities.len()).collect();
        sample.sort_by(|&a, &b| {
            q_values[a]
                .partial_cmp(&q_values[b])
                .expect("q_value to not be NaN")
        });
        sample.truncate(sample_size);
        Ok(sample)
    }
    #[expect(clippy::panic_in_result_fn, reason = "panic implies bug")]
    #[inline]
    fn brewer(&self, rng: &mut R) -> SamplingResult<Vec<usize>> {
        let population_size = self.population_size().get();
        let probabilities = self.probabilities().to_slice_real();
        let eps = self.eps();
        let mut indices = Indices::new(population_size);
        let mut sample = Sample::new(population_size);

        let mut psum: f64 = 0.0;
        for i in (0..population_size).rev() {
            if eps.is_zero(probabilities[i]) {
            } else if eps.is_zero(1.0 - probabilities[i]) {
                sample.add(i);
            } else {
                indices.insert(i);
                psum += probabilities[i];
            }
        }

        if indices.is_empty() {
            return Ok(sample.to_sorted_vec());
        }

        if !eps.difference_is_zero(psum, psum.round()) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }

        let mut rem_sample_size = psum
            .to_usize()
            .expect("probability sum to convert to usize");

        let mut q_probs: Vec<f64> = vec![0.0; probabilities.len()];

        while 0 < rem_sample_size {
            let mut qsum = 0.0;
            let rem_sample_size_f64 = rem_sample_size
                .to_f64()
                .expect("sample_size to convert to f64");

            // Set q_probs
            for &id in indices.list() {
                let p = probabilities[id];
                let q = p * (psum - p) / (psum - p * rem_sample_size_f64);
                q_probs[id] = q;
                qsum += q;
            }

            // Normalize q_probs
            for &id in indices.list() {
                q_probs[id] /= qsum;
                assert!(
                    (0.0..=1.0).contains(&q_probs[id]),
                    "invalid q_probs {} for {}",
                    q_probs[id],
                    id
                );
            }

            // Select unit through pps
            let a_unit = draw(rng, &q_probs);
            indices.remove(a_unit);
            sample.add(a_unit);
            q_probs[a_unit] = 0.0;
            psum -= probabilities[a_unit];
            rem_sample_size -= 1;
        }

        Ok(sample.to_sorted_vec())
    }
    #[inline]
    fn poisson(&self, rng: &mut R) -> Vec<usize> {
        let probabilities = self.probabilities().to_slice_real();
        poisson_internal(rng, probabilities.as_ref())
    }
    #[inline]
    fn conditional_poisson(&self, rng: &mut R, sample_size: usize) -> SamplingResult<Vec<usize>> {
        let probabilities = self.probabilities().to_slice_real();
        let population_size = self.population_size().get();
        if sample_size > population_size {
            return Err(SamplingOptionsError::InvalidSampleSize.into());
        } else if sample_size == 0 {
            return Ok(vec![]);
        } else if sample_size == population_size {
            return Ok((0..population_size).collect::<Vec<usize>>());
        }

        for _ in 0..self.max_iterations().get() {
            let s = poisson_internal(rng, probabilities.as_ref());

            if s.len() == sample_size {
                return Ok(s);
            }
        }

        Err(SamplingError::MaxIterations(self.max_iterations()))
    }
}
