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

//! Probability specifications and containers

use std::borrow::Cow;
use std::iter::repeat_n;
use std::num::{
    NonZeroU128,
    NonZeroUsize,
};

use num_traits::ToPrimitive;

use super::epsilon::Epsilon;
use super::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::number_traits::{
    Number,
    NumberFloat,
};
use crate::probabilities::ProbabilitySet;

/// Interface for constructing probability sets from probability options
pub trait ProbabilityOptions {
    /// The native representation of the probabilities
    type Native: Number;
    /// The real-valued representation of the probabilities
    type Real: NumberFloat;
    /// Returns the native probability set
    fn to_probabilityset(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native>;
    /// Returns a probability set with the real-valued representation.
    fn to_probabilityset_real(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Real>;
    /// Returns the probabilities as a slice in its native representation, and the maximum values
    #[must_use]
    fn to_slice(&self) -> (Cow<'_, [Self::Native]>, Self::Native);
    /// Returns the probabilities as a slice in its real-valued representation
    #[must_use]
    fn to_slice_real(&self) -> Cow<'_, [Self::Real]>;
    /// Returns the population size
    #[must_use]
    fn population_size(&self) -> NonZeroUsize;
    /// Returns the rounded sample size
    #[must_use]
    fn sample_size(&self) -> usize;
    /// Returns the sample size in the real-valued representation
    #[must_use]
    fn sample_size_real(&self) -> Self::Real;
}

/// Probability options for an equal probability design
#[must_use]
pub struct EqualProbabilityOptions {
    /// Population size
    population_size: NonZeroUsize,
    /// Sample size
    sample_size: usize,
}
impl EqualProbabilityOptions {
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
    /// Returns the real-valued probability, i.e. `sample_size / population_size`.
    /// # Panics
    /// Panics if either `sample_size` or `population_size` cannot be converted to `f64`.
    #[must_use]
    #[inline]
    pub fn as_real(&self) -> f64 {
        self.sample_size
            .to_f64()
            .expect("sample size convert to f64")
            / self
                .population_size
                .get()
                .to_f64()
                .expect("population size convert to f64")
    }
}
impl ProbabilityOptions for EqualProbabilityOptions {
    type Native = usize;
    type Real = f64;
    #[inline]
    fn to_probabilityset(&self, _eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
        ProbabilitySet::<Self::Native>::new(
            repeat_n(self.sample_size, self.population_size.get()),
            self.population_size.get(),
        )
    }
    #[inline]
    fn to_probabilityset_real(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Real> {
        ProbabilitySet::<Self::Real>::new(repeat_n(self.as_real(), self.population_size.get()), eps)
    }
    #[inline]
    fn to_slice(&self) -> (Cow<'_, [Self::Native]>, Self::Native) {
        let p = vec![self.sample_size; self.population_size.get()];
        (Cow::Owned(p), self.population_size.get())
    }
    #[inline]
    fn to_slice_real(&self) -> Cow<'_, [Self::Real]> {
        let p = vec![self.as_real(); self.population_size.get()];
        Cow::Owned(p)
    }
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.population_size }
    #[inline]
    fn sample_size(&self) -> usize { self.sample_size }
    #[inline]
    fn sample_size_real(&self) -> Self::Real {
        self.sample_size
            .to_f64()
            .expect("sample size to convert to f64")
    }
}

/// Probability options for an unequal probability design
#[must_use]
pub struct UnequalProbabilityOptions<'bprob, N>
where
    N: Number,
{
    /// Probability data
    data: Cow<'bprob, [N]>,
    /// The maximum value of the probability representation
    max: N,
}
impl<N> UnequalProbabilityOptions<'_, N> where N: Number {}

/// Implements the probability options for float types
macro_rules! prob_opts_impl_float {
    ($t:ty) => {
        impl<'bprob> UnequalProbabilityOptions<'bprob, $t> {
            /// Constructs a new unequal probability specification
            ///
            /// # Errors
            /// Returns an error if the `probabilities` container was empty, or if any provided probability
            /// was not a proper probaility (i.e. within [0.0, 1.0]).
            #[inline]
            pub fn new(probabilities: Cow<'bprob, [$t]>) -> SamplingOptionsResult<Self> {
                let data = probabilities;
                if data.is_empty() {
                    return Err(SamplingOptionsError::InvalidPopulationSize);
                }
                if !data.iter().all(|&p| (0.0..=1.0).contains(&p)) {
                    return Err(SamplingOptionsError::InvalidProbability);
                }
                Ok(Self { data, max: 1.0 })
            }
        }
        impl<'bprob> ProbabilityOptions for UnequalProbabilityOptions<'bprob, $t> {
            type Native = $t;
            type Real = $t;
            #[inline]
            fn to_probabilityset(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
                ProbabilitySet::<Self::Native>::new(self.data.iter().copied(), eps)
            }
            #[inline]
            fn to_probabilityset_real(
                &self,
                eps: Epsilon<Self::Real>,
            ) -> ProbabilitySet<Self::Real> {
                self.to_probabilityset(eps)
            }
            #[inline]
            fn to_slice(&self) -> (Cow<'_, [Self::Native]>, Self::Native) {
                (self.to_slice_real(), 1.0)
            }
            #[inline]
            fn to_slice_real(&self) -> Cow<'_, [Self::Real]> { Cow::Borrowed(self.data.as_ref()) }
            #[inline]
            fn population_size(&self) -> NonZeroUsize {
                NonZeroUsize::new(self.data.len()).expect("data to be non-empty")
            }
            #[inline]
            fn sample_size(&self) -> usize {
                self.sample_size_real()
                    .round()
                    .to_usize()
                    .expect("sample size to convert to usize")
            }
            #[inline]
            fn sample_size_real(&self) -> Self::Real { self.data.iter().sum() }
        }
    };
}
/// Implements the probability options for unsigned integer types
macro_rules! prob_opts_impl_int {
    ($t:ty) => {
        impl<'bprob> UnequalProbabilityOptions<'bprob, $t> {
            /// Constructs a new unequal probability specification
            ///
            /// # Errors
            /// Returns an error if the `probabilities` container was empty, or if any provided probability
            /// was not a proper probaility (i.e. within [0.0, 1.0]).
            #[inline]
            pub fn new(probabilities: Cow<'bprob, [$t]>, max: $t) -> SamplingOptionsResult<Self> {
                let data = probabilities;
                if max <= 0 {
                    return Err(SamplingOptionsError::InvalidEpsilon);
                }
                if data.is_empty() {
                    return Err(SamplingOptionsError::InvalidPopulationSize);
                }
                if !data.iter().all(|&p| (0..=max).contains(&p)) {
                    return Err(SamplingOptionsError::InvalidProbability);
                }
                Ok(Self { data, max })
            }
        }
        impl<'bprob> ProbabilityOptions for UnequalProbabilityOptions<'bprob, $t> {
            type Native = $t;
            type Real = f64;
            #[inline]
            fn to_probabilityset(&self, _eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
                ProbabilitySet::<Self::Native>::new(self.data.iter().copied(), self.max)
            }
            #[inline]
            fn to_probabilityset_real(
                &self,
                eps: Epsilon<Self::Real>,
            ) -> ProbabilitySet<Self::Real> {
                let pop_size = self
                    .population_size()
                    .get()
                    .to_f64()
                    .expect("population size to convert to f64");
                ProbabilitySet::<Self::Real>::new(
                    self.data
                        .iter()
                        .map(|v| v.to_f64().expect("bounded by population size") / pop_size),
                    eps,
                )
            }
            #[inline]
            fn to_slice(&self) -> (Cow<'_, [Self::Native]>, Self::Native) {
                (Cow::Borrowed(self.data.as_ref()), self.max)
            }
            /// # Panics
            /// Panics if max or any internal value does not convert to f64
            #[inline]
            fn to_slice_real(&self) -> Cow<'_, [Self::Real]> {
                let max = self.max.to_f64().expect("max to convert to f64");
                let p = self
                    .data
                    .iter()
                    .map(|v| v.to_f64().expect("v to convert to f64") / max)
                    .collect();
                Cow::Owned(p)
            }
            #[inline]
            fn population_size(&self) -> NonZeroUsize {
                NonZeroUsize::new(self.data.len()).expect("data to be non-empty")
            }
            #[inline]
            fn sample_size(&self) -> usize {
                let s_sum: u128 = self
                    .data
                    .iter()
                    .map(|v| v.to_u128().expect("t to convert to u128"))
                    .sum();
                let u_sum = NonZeroU128::try_from(self.population_size())
                    .expect("population size to convert to U128");
                let ss = s_sum / u_sum;
                let mm = s_sum % u_sum;

                let res = ss.to_usize().expect("s_sum / u_sum to convert to usize");

                if (mm >> 2) < u_sum.get() {
                    res
                } else {
                    res + 1
                }
            }
            #[inline]
            fn sample_size_real(&self) -> Self::Real {
                let s_sum: u128 = self
                    .data
                    .iter()
                    .map(|v| v.to_u128().expect("t to convert to u128"))
                    .sum();
                s_sum.to_f64().expect("u128 to convert to f64")
                    / self
                        .population_size()
                        .get()
                        .to_f64()
                        .expect("population size to convert to f64")
            }
        }
    };
}

prob_opts_impl_int!(usize);
prob_opts_impl_int!(u8);
prob_opts_impl_int!(u16);
prob_opts_impl_int!(u32);
prob_opts_impl_int!(u64);

prob_opts_impl_float!(f64);
