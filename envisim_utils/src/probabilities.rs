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

//! Probability abstractions and container

use num_traits::{
    ConstOne,
    ConstZero,
    ToPrimitive,
};

use crate::kd_tree::searcher::WeightCollection;
use crate::random::Rand;
use crate::utils::{
    ConstructableDataView,
    ContiguousDataView,
    DataView,
    DataViewMut,
    Epsilon,
    Number,
    NumberFloat,
    NumberInt,
    SliceView,
    SliceViewMut,
};

/// Probability context
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct ProbabilityContext<N> {
    /// The maximum value of the probability representations
    max: N,
    /// The epsilon value for comparison between two probability representations
    eps: Epsilon<N>,
}
impl<N> ProbabilityContext<N> {
    /// Constructs a new real-valued context
    #[inline]
    pub fn new_real(eps: Epsilon<N>) -> Self
    where
        N: NumberFloat,
    {
        Self { max: N::ONE, eps }
    }
    /// Constructs a new integer-valued context, given a valid `max`.
    #[inline]
    pub fn new_int(max: N) -> Option<Self>
    where
        N: NumberInt,
    {
        Self::new(max, Epsilon::default())
    }
    /// Constructs a new context, given a valid `max`.
    #[inline]
    pub fn new(max: N, eps: Epsilon<N>) -> Option<Self>
    where
        N: Number,
    {
        max.is_pos_finite().then_some(Self { max, eps })
    }
    /// Returns a reference to `max`
    #[inline]
    pub fn max(&self) -> &N { &self.max }
    /// Returns a reference to `eps`
    #[inline]
    pub fn eps(&self) -> &Epsilon<N> { &self.eps }
}

/// A probability representation
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Probability<N = f64>(N);
impl<N> Probability<N> where N: Number {}

/// A trait for types that can be represented as a probability value
pub trait ProbabilityValue: Copy + PartialOrd + PartialEq {
    /// The inner representation type of the probability
    type N: Number;
    /// Constructs a probability from `prob` if it can be contained `0..=max`.
    #[must_use]
    fn new(prob: Self::N, ctx: &ProbabilityContext<Self::N>) -> Option<Self>
    where
        Self: Sized;
    /// Returns a zero-valued probability.
    #[must_use]
    fn zero() -> Self
    where
        Self: Sized;
    /// Returns a full-valued probability.
    #[must_use]
    fn full(ctx: &ProbabilityContext<Self::N>) -> Self
    where
        Self: Sized;
    /// Returns `true` if `value` is a probability.
    #[must_use]
    #[inline]
    fn is_probability(value: Self::N, max: Self::N) -> bool {
        (Self::N::ZERO..=max).contains(&value)
    }
    /// Returns the inner value
    #[must_use]
    fn get(&self) -> Self::N;
    /// Returns the probability as f64
    #[must_use]
    #[inline]
    fn get_f64(&self) -> Option<f64> { self.get().to_f64() }
    /// Returns `true` if the probability has a zero-value
    #[must_use]
    #[inline]
    fn is_zero(&self, ctx: &ProbabilityContext<Self::N>) -> bool { ctx.eps.is_zero(self.get()) }
    /// Returns `true` if the probability has neither a zero nor a full value
    #[must_use]
    #[inline]
    fn is_partial(&self, ctx: &ProbabilityContext<Self::N>) -> bool {
        !self.is_zero(ctx) && !self.is_full(ctx)
    }
    /// Returns `true` if the probability has a full-value (one)
    #[must_use]
    #[inline]
    fn is_full(&self, ctx: &ProbabilityContext<Self::N>) -> bool {
        self.get() <= ctx.max - ctx.eps.get()
    }
    /// Returns the complement of `self`
    #[must_use]
    fn complement(&self, max: Self::N) -> Self;
    /// Adds `other` to `self`, returning whatever could not be added
    #[must_use]
    fn add(&mut self, other: Self, ctx: &ProbabilityContext<Self::N>) -> Self;
    /// Subtracts `other` from `self`, returning whatever could not be subtracted
    #[must_use]
    fn subtract(&mut self, other: Self) -> Self;
}
impl<N> ProbabilityValue for Probability<N>
where
    N: Number,
{
    type N = N;
    #[inline]
    fn new(prob: Self::N, ctx: &ProbabilityContext<Self::N>) -> Option<Self> {
        Self::is_probability(prob, ctx.max).then_some(Probability(prob))
    }
    #[inline]
    fn zero() -> Self { Probability(N::ZERO) }
    #[inline]
    fn full(ctx: &ProbabilityContext<Self::N>) -> Self { Probability(ctx.max) }
    #[inline]
    fn get(&self) -> Self::N { self.0 }
    /// Returns the complement of `self`
    #[inline]
    fn complement(&self, max: Self::N) -> Self { Probability(max - self.0) }
    #[inline]
    fn add(&mut self, other: Self, ctx: &ProbabilityContext<Self::N>) -> Self {
        let sum = self.0 + other.0;
        if sum <= ctx.max {
            self.0 = sum;
            Probability(N::ZERO)
        } else {
            self.0 = ctx.max;
            Probability(sum - ctx.max)
        }
    }
    #[inline]
    fn subtract(&mut self, other: Self) -> Self {
        if self.0 <= other.0 {
            self.0 = N::ZERO;
            Probability(other.0 - self.0)
        } else {
            self.0 -= other.0;
            Probability(N::ZERO)
        }
    }
}
impl<N> Default for Probability<N>
where
    N: Default,
{
    #[inline]
    fn default() -> Self { Probability(N::default()) }
}
/// A trait for probabilities that can be represented by a real (float) in [0.0, 1.0].
pub trait RealProbabilityValue: ProbabilityValue<N: NumberFloat> {
    /// Constructs a probability from `prob` if it can be contained `0..=max`.
    #[must_use]
    fn new_real(prob: Self::N) -> Option<Self>
    where
        Self: Sized;
    /// Returns `true` if `self` is a probability.
    #[must_use]
    #[inline]
    fn is_real_probability(value: Self::N) -> bool {
        (Self::N::ZERO..=Self::N::ONE).contains(&value)
    }
}
impl<N> RealProbabilityValue for Probability<N>
where
    N: NumberFloat,
{
    #[inline]
    fn new_real(prob: Self::N) -> Option<Self> {
        Self::is_real_probability(prob).then_some(Probability(prob))
    }
}
/// A trait for probability types that can be represented by an integer in [0, MAX], where a proper
/// probability is retrieved by `Self / MAX`.
pub trait IntProbabilityValue: ProbabilityValue<N: NumberInt> {}
impl<N> IntProbabilityValue for Probability<N> where N: NumberInt {}

/// Contains a set of probabilities for some linear population.
#[must_use]
#[derive(Debug, Clone)]
pub struct ProbabilitySet<T, N = <<T as DataView>::Value as ProbabilityValue>::N>
where
    T: DataView<Value: ProbabilityValue<N = N>>,
{
    /// The internal storage for the probability representations
    data: T,
    /// Probability context for the representation
    ctx: ProbabilityContext<N>,
}
impl<T> DataView for ProbabilitySet<T>
where
    T: DataView<Value: ProbabilityValue>,
{
    type Id = T::Id;
    type Value = T::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.data.ids() }
    /// Returns an iterator to the internal data.
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
impl<T> DataViewMut for ProbabilitySet<T>
where
    T: DataViewMut<Value: ProbabilityValue>,
{
    #[inline]
    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Self::Value> {
        self.data.values_mut()
    }
    #[inline]
    fn entries_mut(&mut self) -> impl ExactSizeIterator<Item = (Self::Id, &mut Self::Value)> {
        self.data.entries_mut()
    }
    #[inline]
    fn get_mut(&mut self, id: Self::Id) -> Option<&mut Self::Value> { self.data.get_mut(id) }
}
impl<T> ContiguousDataView for ProbabilitySet<T> where T: ContiguousDataView<Value: ProbabilityValue>
{}
impl<T> SliceView for ProbabilitySet<T>
where
    T: SliceView<Value: ProbabilityValue>,
{
    #[inline]
    fn slice(&self) -> &[Self::Value] { self.data.slice() }
}
impl<T> SliceViewMut for ProbabilitySet<T>
where
    T: SliceViewMut<Value: ProbabilityValue>,
{
    #[inline]
    fn slice_mut(&mut self) -> &mut [Self::Value] { self.data.slice_mut() }
}

impl<T, N> ProbabilitySet<T, N>
where
    T: DataView<Value = Probability<N>>,
    N: Number,
{
    /// Constructs a new set from some data. Fails if data cannot be converted to [`Probability`].
    #[inline]
    #[must_use]
    pub fn from_data<D>(
        data: D,
        ctx: ProbabilityContext<N>,
    ) -> Option<ProbabilitySet<D::ConstructableContainer<Probability<N>>, N>>
    where
        D: ConstructableDataView<Value = N>,
    {
        let mapped_data = D::try_from_iter(
            data.entries()
                .map(|(i, v)| Probability::new(*v, &ctx).ok_or(()).map(|p| (i, p))),
        )
        .ok()?;

        Some(ProbabilitySet {
            data: mapped_data,
            ctx,
        })
    }
}

impl<T, N> ProbabilitySet<T, N>
where
    T: DataView<Value: ProbabilityValue<N = N>>,
    N: Number,
{
    /// Constructs a new set
    #[inline]
    pub fn new(data: T, ctx: ProbabilityContext<<T::Value as ProbabilityValue>::N>) -> Self {
        Self { data, ctx }
    }
    /// Returns a vector of the probabilties contained in the set, as their raw representations
    #[must_use]
    #[inline]
    pub fn to_raw(&self) -> T::ConstructableContainer<N>
    where
        T: ConstructableDataView,
    {
        self.data.iter_map(|(i, v)| (i, v.get()))
    }
    /// Returns a reference to the probability context
    #[inline]
    pub fn ctx(&self) -> &ProbabilityContext<N> { &self.ctx }
    /// Returns `true` if the probability has a zero-value
    #[must_use]
    #[inline]
    pub fn is_zero(&self, id: T::Id) -> Option<bool> { self.get(id).map(|v| v.is_zero(&self.ctx)) }
    /// Returns `true` if the probability has neither a zero nor a full value
    #[must_use]
    #[inline]
    pub fn is_partial(&self, id: T::Id) -> Option<bool> {
        self.get(id).map(|v| v.is_partial(&self.ctx))
    }
    /// Returns `true` if the probability has a full-value (one)
    #[must_use]
    #[inline]
    pub fn is_full(&self, id: T::Id) -> Option<bool> { self.get(id).map(|v| v.is_full(&self.ctx)) }
    /// Adds `value` to the probability of `unit`.
    /// Returns whatever could not be added to `value`.
    #[inline]
    pub fn add(&mut self, id: T::Id, value: T::Value) -> Option<T::Value>
    where
        T: DataViewMut,
    {
        self.data.get_mut(id).map(|v| v.add(value, &self.ctx))
    }
    /// Subtracts `value` from the probability of `unit`.
    /// Returns whatever could not be subtracted from `value`.
    #[inline]
    pub fn subtract(&mut self, id: T::Id, value: T::Value) -> Option<T::Value>
    where
        T: DataViewMut,
    {
        self.data.get_mut(id).map(|v| v.subtract(value))
    }
    /// Sets the probability of unit `id` to `value`.
    /// # Panics
    /// Panics if `value` is not a valid probability representation.
    #[inline]
    pub fn set(&mut self, id: T::Id, value: T::Value) -> Option<()>
    where
        T: DataViewMut,
    {
        self.data.get_mut(id).map(|v| *v = value)
    }
    /// Sets the probability of unit `idx` to the zero representation.
    #[inline]
    pub fn set_zero(&mut self, id: T::Id) -> Option<()>
    where
        T: DataViewMut,
    {
        self.data.get_mut(id).map(|v| *v = T::Value::zero())
    }
    /// Sets the probability of unit `idx` to the full representation.
    #[inline]
    pub fn set_full(&mut self, id: T::Id) -> Option<()>
    where
        T: DataViewMut,
    {
        self.data
            .get_mut(id)
            .map(|v| *v = T::Value::full(&self.ctx))
    }
    /// Draws a random value from the probability representation
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> T::Value
    where
        R: Rand<N>,
    {
        let r = rng.rand_to(self.ctx.max);
        T::Value::new(r, &self.ctx).expect("r < max")
    }
    /// Draws a random value from the probability representation up to `max`.
    /// # Panics
    /// Panics if not `0 < max <= self.max`
    #[inline]
    pub fn draw_partial<R>(&self, rng: &mut R, max: N) -> T::Value
    where
        R: Rand<N>,
    {
        assert!(N::ZERO < max && max <= self.ctx.max, "0 < max <= repr max");
        let r = rng.rand_to(max);
        T::Value::new(r, &self.ctx).expect("r < max")
    }
    /// Returns the weight of `other` on `main`.
    #[inline]
    #[must_use]
    pub fn weight(&self, main: T::Id, other: T::Id) -> Option<f64> {
        self.weight_to(*self.data.get(main)?, other)
    }
    /// Returns the weight of `other` on a probability `prob`.
    /// # Panics
    /// If the probability representations is not convertible to [`f64`]
    #[inline]
    #[must_use]
    pub fn weight_to(&self, prob: T::Value, other: T::Id) -> Option<f64> {
        let max = self.ctx.max.to_f64().expect("max converts to f64");
        let p1 = self
            .data
            .get(other)?
            .get()
            .to_f64()
            .expect("prob converts to f64");
        let w = if prob.is_full(&self.ctx) {
            max - p1
        } else if prob.is_zero(&self.ctx) {
            p1
        } else {
            let p0 = prob.get().to_f64().expect("prob converts to f64");
            if p0 + p1 <= max {
                p1 / (max - p0)
            } else {
                (max - p1) / p0
            }
        };
        Some(w)
    }
}

impl<T> WeightCollection<T::Id> for ProbabilitySet<T>
where
    T: DataView<Value: ProbabilityValue>,
{
    #[inline]
    fn get_weight(&self, id: T::Id) -> Option<f64> {
        self.get(id)
            .map(|v| v.get().to_f64().expect("convert to f64"))
    }
}
