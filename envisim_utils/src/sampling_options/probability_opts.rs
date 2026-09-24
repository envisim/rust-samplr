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

use std::collections::HashMap;
use std::iter::repeat_n;
use std::num::{
    NonZeroU128,
    NonZeroUsize,
};
use std::ops::Range;

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
use crate::utils::{
    ConstructableDataView,
    ContiguousDataView,
    DataView,
    Number,
    NumberFloat,
    NumberInt,
};

/// Interface for constructing probability sets from probability options.
/// `DataView::Value` is the native probability representation.
pub trait ProbabilitiesSpec: DataView<Value: Number> {
    /// The real-valued representation of the probabilities
    type Real: NumberFloat;
    /// Returns the population size
    #[must_use]
    #[inline]
    fn population_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.len()).expect("set to be non-empty")
    }
    /// Returns the rounded sample size.
    #[must_use]
    fn sample_size(&self) -> usize;
    /// Returns the maximum value of the probability representation.
    #[must_use]
    fn max(&self) -> Self::Value;
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
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        let max = self.max_real();
        self.entries().map(move |(id, v)| {
            (
                id,
                <Self::Real as NumCast>::from(*v).expect("native -> real") / max,
            )
        })
    }
    /// Returns the probability at `idx`.
    #[must_use]
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> {
        self.get(id)
            .map(|v| <Self::Real as NumCast>::from(*v).expect("native -> real") / self.max_real())
    }
}
impl<T> ProbabilitiesSpec for &T
where
    T: ProbabilitiesSpec,
{
    type Real = T::Real;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { (**self).population_size() }
    #[inline]
    fn sample_size(&self) -> usize { (**self).sample_size() }
    #[inline]
    fn max(&self) -> Self::Value { (**self).max() }
    #[inline]
    fn population_size_real(&self) -> Self::Real { (**self).population_size_real() }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { (**self).sample_size_real() }
    #[inline]
    fn max_real(&self) -> Self::Real { (**self).max_real() }
    #[inline]
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        (**self).entries_real()
    }
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> { (**self).get_real(id) }
}
impl<T> ProbabilitiesSpec for &mut T
where
    T: ProbabilitiesSpec,
{
    type Real = T::Real;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { (**self).population_size() }
    #[inline]
    fn sample_size(&self) -> usize { (**self).sample_size() }
    #[inline]
    fn max(&self) -> Self::Value { (**self).max() }
    #[inline]
    fn population_size_real(&self) -> Self::Real { (**self).population_size_real() }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { (**self).sample_size_real() }
    #[inline]
    fn max_real(&self) -> Self::Real { (**self).max_real() }
    #[inline]
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        (**self).entries_real()
    }
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> { (**self).get_real(id) }
}

/// Probability options for an equal probability design
#[must_use]
pub struct EqualProbabilities<IDS = Range<usize>> {
    /// Population size
    population_size: NonZeroUsize,
    /// Sample size
    sample_size: usize,
    /// Ids
    ids: IDS,
}
impl EqualProbabilities<Range<usize>> {
    /// Constructs a new equal probability specification over ids `0..population_size`.
    /// # Errors
    /// Returns an error if `sample_size` is not smaller than the `population_size`.
    #[inline]
    pub fn new(
        population_size: NonZeroUsize,
        sample_size: usize,
    ) -> SamplingOptionsResult<EqualProbabilities<Range<usize>>> {
        if population_size.get() < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }
        Ok(Self {
            population_size,
            sample_size,
            ids: 0..population_size.get(),
        })
    }
    /// Constructs a new equal probability specification with a specified range of `ids`.
    /// # Errors
    /// Returns an error if `sample_size` is not smaller than the `population_size`, or if
    /// `ids` is empty.
    #[inline]
    pub fn with_range(
        ids: Range<usize>,
        sample_size: usize,
    ) -> SamplingOptionsResult<EqualProbabilities<Range<usize>>> {
        if !ids.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        let population_size = NonZeroUsize::new(ids.end - ids.start)
            .ok_or(SamplingOptionsError::InvalidPopulationSize)?;
        if population_size.get() < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }
        Ok(Self {
            population_size,
            sample_size,
            ids,
        })
    }
}
impl<ID> EqualProbabilities<ID> {
    /// Constructs a new equal probability specification with some `ids`.
    /// # Errors
    /// Returns an error if `sample_size` is not smaller than the `population_size`, or if
    /// `ids` is empty.
    #[inline]
    pub fn with_ids(ids: ID, sample_size: usize) -> SamplingOptionsResult<EqualProbabilities<ID>>
    where
        ID: DataView,
    {
        let population_size =
            NonZeroUsize::new(ids.len()).ok_or(SamplingOptionsError::InvalidPopulationSize)?;
        if population_size.get() < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }

        Ok(Self {
            population_size,
            sample_size,
            ids,
        })
    }
    /// Returns the real-valued probability, i.e. `sample_size / population_size`.
    /// # Panics
    /// Panics if either `sample_size` or `population_size` cannot be converted to `f64`.
    #[must_use]
    #[inline]
    pub fn as_real(&self) -> f64
    where
        Self: DataView<Value = usize>,
    {
        self.sample_size_real() / self.population_size_real()
    }
}
impl TryFrom<(NonZeroUsize, usize)> for EqualProbabilities<Range<usize>> {
    type Error = SamplingOptionsError;
    #[inline]
    fn try_from(
        (population_size, sample_size): (NonZeroUsize, usize),
    ) -> Result<Self, Self::Error> {
        Self::new(population_size, sample_size)
    }
}
impl TryFrom<(usize, usize)> for EqualProbabilities<Range<usize>> {
    type Error = SamplingOptionsError;
    #[inline]
    fn try_from((population_size, sample_size): (usize, usize)) -> Result<Self, Self::Error> {
        let population_size: NonZeroUsize = NonZeroUsize::new(population_size)
            .ok_or(SamplingOptionsError::InvalidPopulationSize)?;
        Self::new(population_size, sample_size)
    }
}

impl DataView for EqualProbabilities<Range<usize>> {
    type Id = usize;
    type Value = usize;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.ids.clone() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone {
        repeat_n(&self.sample_size, self.population_size.get())
    }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.ids().map(|i| (i, &self.sample_size))
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.ids.contains(&id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> {
        self.contains(id).then_some(&self.sample_size)
    }
    #[inline]
    fn len(&self) -> usize { self.population_size.get() }
    #[inline]
    fn is_empty(&self) -> bool { false }
}
impl<IDS> DataView for EqualProbabilities<IDS>
where
    IDS: DataView,
{
    type Id = IDS::Id;
    type Value = usize;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.ids.ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone {
        repeat_n(&self.sample_size, self.population_size.get())
    }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.ids().map(|i| (i, &self.sample_size))
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.ids.contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> {
        self.contains(id).then_some(&self.sample_size)
    }
    #[inline]
    fn len(&self) -> usize { self.population_size.get() }
    #[inline]
    fn is_empty(&self) -> bool { false }
}
impl<IDS> ContiguousDataView for EqualProbabilities<IDS> where IDS: ContiguousDataView {}
impl ConstructableDataView for EqualProbabilities<Range<usize>> {
    type ConstructableContainer<V> = Box<[V]>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        iter.map(|(_, v)| v).collect()
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        iter.map(|r| r.map(|(_, v)| v)).collect()
    }
}
impl<IDS> ConstructableDataView for EqualProbabilities<IDS>
where
    IDS: DataView,
{
    type ConstructableContainer<V> = HashMap<Self::Id, V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        iter.collect()
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        iter.collect()
    }
}
impl<IDS> ProbabilitiesSpec for EqualProbabilities<IDS>
where
    EqualProbabilities<IDS>: DataView<Value = usize>,
{
    type Real = f64;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.population_size }
    #[inline]
    fn sample_size(&self) -> usize { self.sample_size }
    #[inline]
    fn max(&self) -> Self::Value { self.population_size.get() }
    #[inline]
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        let v = self.as_real();
        self.ids().map(move |i| (i, v))
    }
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> {
        self.contains(id).then_some(self.as_real())
    }
}

/// Stores real-valued probabilities in some slice format
pub struct UnequalProbabilitiesReal<PD>
where
    PD: DataView,
    // PD::Value: NumberFloat,
{
    /// Slicy data
    data: PD,
}
/// Stores integer-valued probabilities in some slice format
pub struct UnequalProbabilitiesInt<PD>
where
    PD: DataView,
    // PD::Value: NumberInt,
{
    /// Slicy data
    data: PD,
    /// Maximum value of the integer-valued probabilities
    max: PD::Value,
}

/// Stores unequal probability options data
pub struct UnequalProbabilities<PO> {
    /// Store
    store: PO,
}
impl<PD> UnequalProbabilities<UnequalProbabilitiesReal<PD>>
where
    PD: DataView,
    PD::Value: NumberFloat,
{
    /// Constructs a new unequal probability specification
    ///
    /// # Errors
    /// Returns an error if the `probabilities` container was empty, or if any provided probability
    /// was not a proper probaility (i.e. within [0.0, 1.0]).
    #[inline]
    pub fn new(probabilities: PD) -> SamplingOptionsResult<Self> {
        if probabilities.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !probabilities
            .values()
            .all(|&p| (PD::Value::ZERO..=PD::Value::ONE).contains(&p))
        {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        Ok(Self {
            store: UnequalProbabilitiesReal {
                data: probabilities,
            },
        })
    }
}
impl<PD> UnequalProbabilities<UnequalProbabilitiesInt<PD>>
where
    PD: DataView,
    PD::Value: NumberInt,
{
    /// Constructs a new unequal probability specification
    ///
    /// # Errors
    /// Returns an error if
    /// - the `probabilities` container was empty,
    /// - any provided probability was not a proper probaility (i.e. within [0.0, 1.0]), or
    /// - `max` is zero.
    #[inline]
    pub fn new_int(probabilities: PD, max: PD::Value) -> SamplingOptionsResult<Self> {
        if !max.is_pos_finite() {
            return Err(SamplingOptionsError::InvalidProbability);
        } else if probabilities.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !probabilities
            .values()
            .all(|&p| (PD::Value::ZERO..=max).contains(&p))
        {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        Ok(Self {
            store: UnequalProbabilitiesInt {
                data: probabilities,
                max,
            },
        })
    }
}

impl<PD> DataView for UnequalProbabilitiesReal<PD>
where
    PD: DataView,
{
    type Id = PD::Id;
    type Value = PD::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.data.ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { self.data.values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.data.entries()
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.data.contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { self.data.get(id) }
    #[inline]
    fn len(&self) -> usize { self.data.len() }
    #[inline]
    fn is_empty(&self) -> bool { self.data.is_empty() }
}
impl<PD> DataView for UnequalProbabilitiesInt<PD>
where
    PD: DataView<Value: NumberInt>,
{
    type Id = PD::Id;
    type Value = PD::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.data.ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { self.data.values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.data.entries()
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.data.contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { self.data.get(id) }
    #[inline]
    fn len(&self) -> usize { self.data.len() }
    #[inline]
    fn is_empty(&self) -> bool { self.data.is_empty() }
}
impl<PO> DataView for UnequalProbabilities<PO>
where
    PO: DataView,
{
    type Id = PO::Id;
    type Value = PO::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.store.ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { self.store.values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.store.entries()
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.store.contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { self.store.get(id) }
    #[inline]
    fn len(&self) -> usize { self.store.len() }
    #[inline]
    fn is_empty(&self) -> bool { self.store.is_empty() }
}

impl<PD> ConstructableDataView for UnequalProbabilitiesReal<PD>
where
    PD: ConstructableDataView,
{
    type ConstructableContainer<V> = PD::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        PD::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        PD::try_from_iter(iter)
    }
}
impl<PD> ConstructableDataView for UnequalProbabilitiesInt<PD>
where
    PD: ConstructableDataView<Value: NumberInt>,
{
    type ConstructableContainer<V> = PD::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        PD::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        PD::try_from_iter(iter)
    }
}
impl<PO> ConstructableDataView for UnequalProbabilities<PO>
where
    PO: ConstructableDataView<Value: Number>,
{
    type ConstructableContainer<V> = PO::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        PO::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        PO::try_from_iter(iter)
    }
}

impl<PD> ProbabilitiesSpec for UnequalProbabilitiesReal<PD>
where
    PD: DataView<Value: NumberFloat>,
{
    type Real = PD::Value;
    #[inline]
    fn sample_size(&self) -> usize {
        self.sample_size_real()
            .round()
            .to_usize()
            .expect("sample size to convert to usize")
    }
    #[inline]
    fn max(&self) -> Self::Value { <Self::Value>::ONE }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { self.values().copied().sum() }
    #[inline]
    fn max_real(&self) -> Self::Real { self.max() }
    #[inline]
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        self.entries().map(|(i, v)| (i, *v))
    }
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> { self.get(id).copied() }
}
impl<PD> ProbabilitiesSpec for UnequalProbabilitiesInt<PD>
where
    PD: DataView<Value: NumberInt>,
{
    type Real = f64;
    #[inline]
    fn sample_size(&self) -> usize {
        let s_sum: u128 = self
            .values()
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
    fn max(&self) -> Self::Value { self.max }
    #[inline]
    fn sample_size_real(&self) -> Self::Real {
        let s_sum: u128 = self
            .values()
            .map(|v| v.to_u128().expect("t to convert to u128"))
            .sum();
        <Self::Real as NumCast>::from(s_sum).expect("u128 to convert to real")
            / self.population_size_real()
    }
}
impl<PO> ProbabilitiesSpec for UnequalProbabilities<PO>
where
    PO: ProbabilitiesSpec,
{
    type Real = PO::Real;
    #[inline]
    fn population_size(&self) -> NonZeroUsize { self.store.population_size() }
    #[inline]
    fn sample_size(&self) -> usize { self.store.sample_size() }
    #[inline]
    fn max(&self) -> Self::Value { self.store.max() }
    #[inline]
    fn population_size_real(&self) -> Self::Real { self.store.population_size_real() }
    #[inline]
    fn sample_size_real(&self) -> Self::Real { self.store.sample_size_real() }
    #[inline]
    fn max_real(&self) -> Self::Real { self.store.max_real() }
    #[inline]
    fn entries_real(&self) -> impl ExactSizeIterator<Item = (Self::Id, Self::Real)> + Clone {
        self.store.entries_real()
    }
    #[inline]
    fn get_real(&self, id: Self::Id) -> Option<Self::Real> { self.store.get_real(id) }
}
