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

//! Unequal probability sampling designs.
//!
//! Implements [`UnequalProbabilitySampling`] for [`SamplingOptions`].
//!
//! # References
//! Rosén, B. (2000).
//! A user’s guide to Pareto pi-ps sampling. R & D Report 2000:6.
//! Stockholm: Statistiska Centralbyrån.

use envisim_utils::indices::Indices;
use envisim_utils::random::{
    Rand,
    Rng,
};
use envisim_utils::sample::Sample;
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    BaseProbabilitiesSpec,
    ProbabilitiesSpec,
    SamplingOptionsError,
    SamplingOptionsRng,
};
use envisim_utils::utils::{
    DataView,
    DataViewMut,
    Epsilon,
};
use num_traits::{
    ConstZero,
    ToPrimitive,
    Zero,
};

pub use crate::error::SamplingError;
use crate::error::SamplingResult;
use crate::utils::poisson_internal;

/// Draws a single unit using pps
/// Assumes probabilites sum to 1.0
#[must_use]
#[inline]
fn draw<'bprob, R, ID, I>(rng: &mut R, probabilities: I) -> Option<ID>
where
    R: Rand<f64>,
    ID: Copy,
    I: ExactSizeIterator<Item = (ID, &'bprob f64)>,
{
    let rv = rng.rand();
    let mut pacc: f64 = 0.0;
    let mut outer_id: Option<ID> = None;

    for (id, &p) in probabilities {
        outer_id = Some(id);
        pacc += p;

        if rv <= pacc {
            break;
        }
    }

    outer_id
}

/// Draws a single unit using pps
/// Assumes probabilites sum to psum
#[must_use]
#[inline]
fn draw3<R, PS>(rng: &mut R, probs: PS, psum: PS::Value) -> Option<PS::Id>
where
    R: Rand<PS::Value>,
    PS: BaseProbabilitiesSpec,
{
    let rv = rng.rand_to(psum);
    let mut pacc = PS::Value::ZERO;
    let mut outer_id: Option<PS::Id> = None;

    for (id, &p) in probs.entries() {
        outer_id = Some(id);
        pacc += p;

        if rv <= pacc {
            break;
        }
    }

    outer_id
}

/// Provides simple unequal probability sampling methods.
pub trait UnequalProbabilitySampling<ID, R>
where
    R: Rng,
{
    /// Draw a with replacment sample according to draw probabilities.
    /// Probabilities must sum to 1.0.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.1; 10];
    /// let s = SamplingOptions::new(p)?.with_replacement(&mut rng, 5)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to 1.0.
    fn with_replacement(&self, rng: &mut R, n: usize) -> SamplingResult<Vec<ID>>;
    /// Draw a sample using a sampford design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.sampford(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn sampford(&self, rng: &mut R) -> SamplingResult<Vec<ID>>;
    /// Draw a sample using a pareto design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.pareto(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn pareto(&self, rng: &mut R) -> SamplingResult<Vec<ID>>;
    /// Draw a sample using a brewer design.
    /// Probabilities must sum to an integer.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.brewer(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if probabilities does not sum to an integer.
    fn brewer(&self, rng: &mut R) -> SamplingResult<Vec<ID>>;
    /// Draw a sample using a poisson design.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.poisson(&mut rng);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn poisson(&self, rng: &mut R) -> Vec<ID>;
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
    /// let s = SamplingOptions::new(p)?.conditional_poisson(&mut rng, 5)?;
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if `sample_size` is larger than the population size.
    fn conditional_poisson(&self, rng: &mut R, sample_size: usize) -> SamplingResult<Vec<ID>>;
}
impl<R, PO, AUX, BAL> UnequalProbabilitySampling<PO::Id, R> for SamplingOptions<PO, AUX, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64>,
{
    #[inline]
    fn with_replacement(&self, rng: &mut R, n: usize) -> SamplingResult<Vec<PO::Id>> {
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

        let mut sample = Vec::<PO::Id>::with_capacity(n);
        let mut psum: f64 = 0.0;
        let mut rv_iter = rvs.iter();
        let mut rv = *rv_iter.next().expect("at least one rv to exist");

        // Add units for which rv is in [psum, psum+p)
        // Go up one p when psum+p < rv
        // Go up one rv when sample has been pushed
        'outer: for (id, p) in self.probabilities().entries_real() {
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
                        }
                        _ => break 'outer,
                    }
                }
            }
        }

        Ok(sample)
    }
    #[inline]
    fn sampford(&self, rng: &mut R) -> SamplingResult<Vec<PO::Id>> {
        let psum: PO::Value = self.probabilities().values().copied().sum();

        if psum.is_zero() {
            return Ok(vec![]);
        } else if psum == self.probabilities().max() {
            return Ok(vec![draw3(rng, self.probabilities(), psum).expect("N > 0")]);
        }

        let rest = psum % self.probabilities().max();
        let sample_size = self.probabilities().sample_size();

        // If rest is non-zero, sample size is non-integer
        if !Epsilon::<PO::Value>::default().is_zero(rest) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }

        for _ in 0..self.max_iterations().get() {
            let mut sample = poisson_internal(rng, self.probabilities());
            if sample.len() != sample_size - 1 {
                continue;
            }

            let a_unit = draw3(rng, self.probabilities(), psum).expect("N > 0");

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
    fn pareto(&self, rng: &mut R) -> SamplingResult<Vec<PO::Id>> {
        let psum: PO::Value = self.probabilities().values().copied().sum();

        if psum.is_zero() {
            return Ok(vec![]);
        } else if psum == self.probabilities().max() {
            return Ok(vec![draw3(rng, self.probabilities(), psum).expect("N > 0")]);
        }

        let rest = psum % self.probabilities().max();
        let sample_size = self.probabilities().sample_size();

        // If rest is non-zero, sample size is non-integer
        if !Epsilon::<PO::Value>::default().is_zero(rest) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }

        let eps = self.eps();
        let mut q_values: Vec<(PO::Id, f64)> = self
            .probabilities()
            .entries_real()
            .map(|(id, p)| {
                let u: f64 = rng.rand();

                if eps.is_zero(p) || eps.is_zero(1.0 - u) {
                    return (id, f64::INFINITY);
                }

                let res = (u * (1.0 - p)) / (p * (1.0 - u));

                if res.is_nan() {
                    return (id, f64::INFINITY);
                }

                (id, res)
            })
            .collect();
        q_values.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("q_value to not be NaN"));
        Ok(q_values
            .iter()
            .map(|(id, _)| *id)
            .take(sample_size)
            .collect())
    }
    #[expect(clippy::panic_in_result_fn, reason = "panic implies bug")]
    #[inline]
    fn brewer(&self, rng: &mut R) -> SamplingResult<Vec<PO::Id>> {
        let psum: PO::Value = self.probabilities().values().copied().sum();

        if psum.is_zero() {
            return Ok(vec![]);
        } else if psum == self.probabilities().max() {
            return Ok(vec![draw3(rng, self.probabilities(), psum).expect("N > 0")]);
        }

        let rest = psum % self.probabilities().max();
        let sample_size = self.probabilities().sample_size();

        // If rest is non-zero, sample size is non-integer
        if !Epsilon::<PO::Value>::default().is_zero(rest) {
            return Err(SamplingError::IncorrectProbabilitiesIntegerSum);
        }

        let population_size = self.population_size();
        let mut indices = Indices::new(population_size.get());
        let mut sample = Sample::new(population_size.get());

        let eps = self.eps();
        for (id, p) in self.probabilities().entries_real() {
            if eps.is_zero(p) {
            } else if eps.is_zero(1.0 - p) {
                sample.add(id);
            } else {
                indices.insert(id);
            }
        }

        if indices.is_empty() {
            return Ok(sample.to_sorted_vec());
        }

        let mut psum_f64 = self.probabilities().sample_size_real();
        let mut rem_sample_size = sample_size;

        let mut q_probs = self.probabilities().iter_map(|(id, _)| (id, 0.0));

        while 0 < rem_sample_size {
            let mut qsum = 0.0;
            let rem_sample_size_f64 = rem_sample_size
                .to_f64()
                .expect("sample_size to convert to f64");

            // Set q_probs
            for &id in indices.list() {
                let p = self.probabilities().get_real(id).expect("id to exist");
                let q = p * (psum_f64 - p) / (psum_f64 - p * rem_sample_size_f64);
                *q_probs.get_mut(id).expect("id to exist") = q;
                qsum += q;
            }

            // Normalize q_probs
            for &id in indices.list() {
                let q = q_probs.get_mut(id).expect("id to exist");
                *q /= qsum;
                assert!((0.0..=1.0).contains(q), "invalid q_probs {q} for {id:?}");
            }

            // Select unit through pps
            let a_unit = draw(rng, q_probs.entries()).expect("N > 0");
            indices.remove(a_unit);
            sample.add(a_unit);
            *q_probs.get_mut(a_unit).expect("id to exist") = 0.0;
            psum_f64 -= self
                .probabilities()
                .get_real(a_unit)
                .expect("a_unit to exist");
            rem_sample_size -= 1;
        }

        Ok(sample.to_sorted_vec())
    }
    #[inline]
    fn poisson(&self, rng: &mut R) -> Vec<PO::Id> { poisson_internal(rng, self.probabilities()) }
    #[inline]
    fn conditional_poisson(&self, rng: &mut R, sample_size: usize) -> SamplingResult<Vec<PO::Id>> {
        let population_size = self.population_size().get();
        if sample_size > population_size {
            return Err(SamplingOptionsError::InvalidSampleSize.into());
        } else if sample_size == 0 {
            return Ok(vec![]);
        } else if sample_size == population_size {
            return Ok(self.probabilities().ids().collect());
        }

        for _ in 0..self.max_iterations().get() {
            let s = poisson_internal(rng, self.probabilities());

            if s.len() == sample_size {
                return Ok(s);
            }
        }

        Err(SamplingError::MaxIterations(self.max_iterations()))
    }
}
