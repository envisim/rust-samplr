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

use std::iter::repeat_n;
use std::num::{
    NonZeroU128,
    NonZeroUsize,
};

use num_integer::Integer;
use num_traits::{
    ConstOne,
    ConstZero,
    Float,
    NumCast,
    ToPrimitive,
};

use super::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::Epsilon;
use crate::matrix::RawData;
use crate::number_traits::{
    Number,
    NumberFloat,
    NumberInt,
};
use crate::probabilities::ProbabilitySet;

/// Interface for constructing probability sets from probability options
pub trait ProbabilityOptions {
    /// The native representation of the probabilities
    type Native: Number;
    /// The real-valued representation of the probabilities
    type Real: NumberFloat;
    /// Returns the population size
    #[must_use]
    fn population_size(&self) -> NonZeroUsize;
    /// Returns the rounded sample size.
    #[must_use]
    fn sample_size(&self) -> usize;
    /// Returns the maximum value of the probability representation.
    #[must_use]
    fn max(&self) -> Self::Native;
    // /// Returns the epsilon value of the native-valued representation.
    // fn eps(&self) -> Epsilon<Self::Native> { Epsilon::default() }
    /// Returns an iterator to the probabilities in its native representation.
    #[must_use]
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::Native> + Clone;
    /// Returns the probability set
    #[inline]
    fn to_probabilityset(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
        let eps_v = <Self::Native as NumCast>::from(eps.get()).expect("eps real -> native");
        let eps_c = Epsilon::new(eps_v).expect("constructable eps");
        ProbabilitySet::try_new(self.iter(), self.max(), eps_c)
            .expect("ProbabilitySet to be constructable")
    }
    /// Returns the population size in the real-valued representation.
    #[must_use]
    #[inline]
    fn population_size_real(&self) -> Self::Real {
        <Self::Real as NumCast>::from(self.population_size().get())
            .expect("population size -> real")
    }
    /// Returns the sample size in the real-valued representation.
    #[must_use]
    #[inline]
    fn sample_size_real(&self) -> Self::Real {
        <Self::Real as NumCast>::from(self.sample_size()).expect("sample size -> real")
    }
    /// Returns the maximum value in the real-valued representation.
    #[must_use]
    #[inline]
    fn max_real(&self) -> Self::Real {
        <Self::Real as NumCast>::from(self.max()).expect("max -> real")
    }
    /// Returns an iterator to the probabilities in its real-valued representation.
    #[must_use]
    #[inline]
    fn iter_real(&self) -> impl ExactSizeIterator<Item = Self::Real> + Clone {
        self.iter()
            .map(|v| <Self::Real as NumCast>::from(v).expect("native -> real") / self.max_real())
    }
    /// Returns a probability set with the real-valued representation.
    #[inline]
    fn to_probabilityset_real(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Real> {
        ProbabilitySet::try_new(self.iter_real(), Self::Real::ONE, eps)
            .expect("ProbabilitySet to be constructable")
    }
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
    pub fn as_real(&self) -> f64 { self.sample_size_real() / self.population_size_real() }
}
impl ProbabilityOptions for EqualProbabilityOptions {
    type Native = usize;
    type Real = f64;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.population_size }
    #[inline]
    fn sample_size(&self) -> usize { self.sample_size }
    #[inline]
    fn max(&self) -> Self::Native { self.population_size.get() }
    #[inline]
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::Native> + Clone {
        repeat_n(self.sample_size, self.population_size.get())
    }
    #[inline]
    fn iter_real(&self) -> impl ExactSizeIterator<Item = Self::Real> + Clone {
        repeat_n(self.as_real(), self.population_size.get())
    }
}

/// Stores real-valued probabilities in some slice format
pub struct RealUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberFloat,
{
    /// Slicy data
    data: PD,
}
/// Stores integer-valued probabilities in some slice format
pub struct IntUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberInt,
{
    /// Slicy data
    data: PD,
    /// Maximum value of the integer-valued probabilities
    max: PD::Elem,
}

/// Access the internal data of an unequal probability options
pub trait UnequalProbabilityOptionsAccess: ProbabilityOptions {
    /// Returns a slice of the internal data
    #[must_use]
    fn as_slice(&self) -> &[<Self as ProbabilityOptions>::Native];
}
/// Stores unequal probability options data
pub struct UnequalProbabilityOptions<PO> {
    /// Store
    store: PO,
}
impl<PD> UnequalProbabilityOptions<RealUnequalProbabilityOptions<PD>>
where
    PD: RawData,
    PD::Elem: NumberFloat,
{
    /// Constructs a new unequal probability specification
    ///
    /// # Errors
    /// Returns an error if the `probabilities` container was empty, or if any provided probability
    /// was not a proper probaility (i.e. within [0.0, 1.0]).
    #[inline]
    pub fn new(probabilities: PD) -> SamplingOptionsResult<Self> {
        let data = probabilities.data();
        if data.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !data
            .iter()
            .all(|&p| (PD::Elem::ZERO..=PD::Elem::ONE).contains(&p))
        {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        Ok(Self {
            store: RealUnequalProbabilityOptions {
                data: probabilities,
            },
        })
    }
}
impl<PD> UnequalProbabilityOptions<IntUnequalProbabilityOptions<PD>>
where
    PD: RawData,
    PD::Elem: NumberInt,
{
    /// Constructs a new unequal probability specification
    ///
    /// # Errors
    /// Returns an error if
    /// - the `probabilities` container was empty,
    /// - any provided probability was not a proper probaility (i.e. within [0.0, 1.0]), or
    /// - `max` is zero.
    #[inline]
    pub fn new_int(probabilities: PD, max: PD::Elem) -> SamplingOptionsResult<Self> {
        let data = probabilities.data();
        if !max.is_pos_finite() {
            return Err(SamplingOptionsError::InvalidProbability);
        } else if data.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !data.iter().all(|&p| (PD::Elem::ZERO..=max).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        Ok(Self {
            store: IntUnequalProbabilityOptions {
                data: probabilities,
                max,
            },
        })
    }
}

impl<PD> UnequalProbabilityOptionsAccess for RealUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberFloat,
{
    #[inline]
    fn as_slice(&self) -> &[<Self as ProbabilityOptions>::Native] { self.data.data() }
}
impl<PD> UnequalProbabilityOptionsAccess for IntUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberInt,
{
    #[inline]
    fn as_slice(&self) -> &[<Self as ProbabilityOptions>::Native] { self.data.data() }
}
impl<PO> UnequalProbabilityOptionsAccess for UnequalProbabilityOptions<PO>
where
    PO: UnequalProbabilityOptionsAccess,
{
    #[inline]
    fn as_slice(&self) -> &[<Self as ProbabilityOptions>::Native] { self.store.as_slice() }
}

impl<PO> ProbabilityOptions for UnequalProbabilityOptions<PO>
where
    PO: ProbabilityOptions,
{
    type Native = PO::Native;
    type Real = PO::Real;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.store.population_size() }
    #[inline]
    fn sample_size(&self) -> usize { self.store.sample_size() }
    #[inline]
    fn max(&self) -> Self::Native { self.store.max() }
    #[inline]
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::Native> + Clone { self.store.iter() }
    #[inline]
    fn to_probabilityset(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
        self.store.to_probabilityset(eps)
    }
    #[inline]
    fn population_size_real(&self) -> Self::Real { self.store.population_size_real() }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { self.store.sample_size_real() }
    #[inline]
    fn max_real(&self) -> Self::Real { self.store.max_real() }
    #[inline]
    fn iter_real(&self) -> impl ExactSizeIterator<Item = Self::Real> + Clone {
        self.store.iter_real()
    }
    #[inline]
    fn to_probabilityset_real(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Real> {
        self.store.to_probabilityset_real(eps)
    }
}
impl<PD> ProbabilityOptions for RealUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberFloat,
{
    type Native = PD::Elem;
    type Real = PD::Elem;
    #[inline]
    fn population_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.data.data().len()).expect("data to be non-empty")
    }
    #[inline]
    fn sample_size(&self) -> usize {
        self.sample_size_real()
            .round()
            .to_usize()
            .expect("sample size to convert to usize")
    }
    #[inline]
    fn max(&self) -> Self::Native { <Self::Native>::ONE }
    #[inline]
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::Native> + Clone {
        self.data.data().iter().copied()
    }
    #[inline]
    fn to_probabilityset(&self, eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
        ProbabilitySet::try_new(self.iter(), self.max(), eps)
            .expect("ProbabilitySet to be constructable")
    }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { self.iter_real().sum() }
    #[inline]
    fn iter_real(&self) -> impl ExactSizeIterator<Item = Self::Real> + Clone { self.iter() }
}

impl<PD> ProbabilityOptions for IntUnequalProbabilityOptions<PD>
where
    PD: RawData,
    PD::Elem: NumberInt,
{
    type Native = PD::Elem;
    type Real = f64;
    #[inline]
    fn population_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.data.data().len()).expect("data to be non-empty")
    }
    #[inline]
    fn sample_size(&self) -> usize {
        let s_sum: u128 = self
            .iter()
            .map(|v| v.to_u128().expect("t to convert to u128"))
            .sum();
        let u_sum = NonZeroU128::try_from(self.population_size())
            .expect("population size to convert to U128");
        let (ss, mm) = s_sum.div_mod_floor(&u_sum.get());

        let res = ss.to_usize().expect("s_sum / u_sum to convert to usize");

        if (mm << 1) < u_sum.get() {
            res
        } else {
            res + 1
        }
    }
    #[inline]
    fn max(&self) -> Self::Native { self.max }
    #[inline]
    fn iter(&self) -> impl ExactSizeIterator<Item = Self::Native> + Clone {
        self.data.data().iter().copied()
    }
    #[inline]
    fn to_probabilityset(&self, _eps: Epsilon<Self::Real>) -> ProbabilitySet<Self::Native> {
        ProbabilitySet::try_new(self.iter(), self.max(), Epsilon::default())
            .expect("ProbabilitySet to be constructable")
    }
    #[inline]
    fn sample_size_real(&self) -> Self::Real {
        let s_sum: u128 = self
            .iter()
            .map(|v| v.to_u128().expect("t to convert to u128"))
            .sum();
        <Self::Real as NumCast>::from(s_sum).expect("u128 to convert to real")
            / self.population_size_real()
    }
    #[inline]
    fn iter_real(&self) -> impl ExactSizeIterator<Item = Self::Real> + Clone {
        self.iter()
            .map(|v| <Self::Real as NumCast>::from(v).expect("native -> real") / self.max_real())
    }
}
