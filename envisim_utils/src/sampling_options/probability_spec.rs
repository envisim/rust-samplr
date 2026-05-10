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

//! Probability specifications and/or containers

use std::borrow::Cow;
use std::num::NonZeroUsize;

use num_traits::ToPrimitive;

use super::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::probabilities::{
    ExactProbabilities,
    FloatProbabilities,
    ProbabilityStore,
};

// Probability specification
pub trait ProbabilitySpec {
    type Native: ProbabilityStore;
    #[must_use]
    fn population_size(&self) -> NonZeroUsize;
    #[expect(clippy::unwrap_used, reason = "usize to f64 conversion")]
    #[must_use]
    #[inline]
    fn population_size_f64(&self) -> f64 { self.population_size().get().to_f64().unwrap() }
    #[must_use]
    fn sample_size(&self) -> usize;
    #[expect(clippy::unwrap_used, reason = "usize to f64 conversion")]
    #[must_use]
    #[inline]
    fn sample_size_f64(&self) -> f64 { self.sample_size().to_f64().unwrap() }
    /// Returns probabilities as f64 slice
    #[must_use]
    fn as_f64_slice(&self) -> Cow<'_, [f64]>;
    /// Returns probabilities as f64 slice
    #[must_use]
    #[inline]
    fn as_equal(&self) -> Option<ProbabilitySpecEqual> { None }

    #[must_use]
    fn to_native(&self, eps: f64) -> Self::Native;
    fn to_float(&self, eps: f64) -> FloatProbabilities;
}

#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct ProbabilitySpecEqual {
    /// Population size
    population_size: NonZeroUsize,
    /// Sample size
    sample_size: usize,
}
impl ProbabilitySpecEqual {
    /// Constructs a new equal probability specification
    ///
    /// # Errors
    /// Returns an error if population size cannot be converted into a [`NonZeroUsize`], or if the
    /// sample size is larger than the population size.
    #[inline]
    pub fn new<NZ>(population_size: NZ, sample_size: usize) -> SamplingOptionsResult<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let population_size: NonZeroUsize = population_size
            .try_into()
            .map_err(|_| SamplingOptionsError::InvalidPopulationSize)?;
        if population_size.get() < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }
        Ok(Self {
            population_size,
            sample_size,
        })
    }
    #[must_use]
    #[inline]
    pub fn as_f64(&self) -> f64 { self.sample_size_f64() / self.population_size_f64() }
}
impl ProbabilitySpec for ProbabilitySpecEqual {
    type Native = ExactProbabilities;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.population_size }
    #[inline]
    fn sample_size(&self) -> usize { self.sample_size }
    #[inline]
    fn as_f64_slice(&self) -> Cow<'_, [f64]> {
        Cow::Owned(vec![self.as_f64(); self.population_size.get()])
    }
    #[inline]
    fn as_equal(&self) -> Option<ProbabilitySpecEqual> { Some(*self) }
    #[inline]
    fn to_native(&self, _eps: f64) -> Self::Native { ExactProbabilities::new_equal(*self) }
    #[inline]
    fn to_float(&self, eps: f64) -> FloatProbabilities {
        FloatProbabilities::new(self.as_f64_slice().into_owned(), eps)
    }
}

#[must_use]
#[derive(Clone, Debug)]
pub struct ProbabilitySpecUnequal<'bprob> {
    /// Probability data
    data: Cow<'bprob, [f64]>,
}
impl ProbabilitySpec for ProbabilitySpecUnequal<'_> {
    type Native = FloatProbabilities;
    #[inline]
    fn population_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.data.len()).expect("data to not be empty")
    }
    #[inline]
    fn sample_size_f64(&self) -> f64 { self.data.iter().sum::<f64>() }
    #[inline]
    fn sample_size(&self) -> usize {
        self.sample_size_f64()
            .round()
            .to_usize()
            .expect("sample size to be convertable to usize")
    }
    #[inline]
    fn as_f64_slice(&self) -> Cow<'_, [f64]> { Cow::Borrowed(self.data.as_ref()) }
    #[inline]
    fn to_native(&self, eps: f64) -> Self::Native {
        FloatProbabilities::new(self.data.to_vec(), eps)
    }
    #[inline]
    fn to_float(&self, eps: f64) -> FloatProbabilities { Self::to_native(self, eps) }
}
impl<'bprob> ProbabilitySpecUnequal<'bprob> {
    /// Constructs a new unequal probability specification
    ///
    /// # Errors
    /// Returns an error if the `probabilities` container was empty, or if any provided probability
    /// was not a proper probaility (i.e. within [0.0, 1.0]).
    #[inline]
    pub fn new(probabilities: Cow<'bprob, [f64]>) -> SamplingOptionsResult<Self> {
        let data = probabilities;
        if data.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        if !data.iter().all(|&p| (0.0..=1.0).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        Ok(Self { data })
    }
}
