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
    ConstZero,
    NumCast,
    ToPrimitive,
};

use crate::kd_tree::searcher::WeightCollection;
use crate::random::Rand;
use crate::sampling_options::ProbabilitiesSpec;
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
#[must_use]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Probability<N = f64>(N);
impl<N> Probability<N>
where
    N: Number,
{
    /// Constructs a probability from `prob` if it can be contained `0..=max`.
    #[must_use]
    #[inline]
    pub fn new(prob: N, ctx: &ProbabilityContext<N>) -> Option<Self> {
        Self::is_probability(prob, ctx).then_some(Probability(prob))
    }
    /// Constructs a probability from `prob` if it can be contained `0..=max`.
    #[must_use]
    #[inline]
    pub fn new_real(prob: N) -> Option<Self>
    where
        N: NumberFloat,
    {
        Self::is_real_probability(prob).then_some(Probability(prob))
    }
    /// Returns a zero-valued probability.
    #[inline]
    pub fn zero() -> Self { Probability(N::ZERO) }
    /// Returns a full-valued probability.
    #[inline]
    pub fn full(ctx: &ProbabilityContext<N>) -> Self { Probability(ctx.max) }
    /// Returns `true` if `value` is a probability.
    #[must_use]
    #[inline]
    pub fn is_probability(value: N, ctx: &ProbabilityContext<N>) -> bool {
        (N::ZERO..=ctx.max).contains(&value)
    }
    /// Returns `true` if `self` is a probability.
    #[must_use]
    #[inline]
    pub fn is_real_probability(value: N) -> bool
    where
        N: NumberFloat,
    {
        (N::ZERO..=N::ONE).contains(&value)
    }
    /// Returns the inner value
    #[must_use]
    #[inline]
    pub fn get(&self) -> N { self.0 }
    /// Returns the probability as f64
    #[must_use]
    #[inline]
    pub fn get_f64(&self) -> Option<f64> { self.get().to_f64() }
    /// Returns `true` if the probability has a zero-value
    #[must_use]
    #[inline]
    pub fn is_zero(&self, ctx: &ProbabilityContext<N>) -> bool { ctx.eps.is_zero(self.get()) }
    /// Returns `true` if the probability has neither a zero nor a full value
    #[must_use]
    #[inline]
    pub fn is_partial(&self, ctx: &ProbabilityContext<N>) -> bool {
        !self.is_zero(ctx) && !self.is_full(ctx)
    }
    /// Returns `true` if the probability has a full-value (one)
    #[must_use]
    #[inline]
    pub fn is_full(&self, ctx: &ProbabilityContext<N>) -> bool {
        ctx.max - ctx.eps.get() <= self.get()
    }
    /// Returns the complement of `self`
    #[inline]
    pub fn complement(&self, ctx: &ProbabilityContext<N>) -> Self { Probability(ctx.max - self.0) }
    /// Adds `other` to `self`, returning whatever could not be added
    #[inline]
    pub fn add(&mut self, other: Self, ctx: &ProbabilityContext<N>) -> Self {
        let sum = self.0 + other.0;
        if sum <= ctx.max {
            self.0 = sum;
            Probability(N::ZERO)
        } else {
            self.0 = ctx.max;
            Probability(sum - ctx.max)
        }
    }
    /// Subtracts `other` from `self`, returning whatever could not be subtracted
    #[inline]
    pub fn subtract(&mut self, other: Self) -> Self {
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

/// Contains a set of probabilities for some linear population.
#[must_use]
#[derive(Debug, Clone)]
pub struct ProbabilitySet<T, N> {
    /// The internal storage for the probability representations
    data: T,
    /// Probability context for the representation
    ctx: ProbabilityContext<N>,
}
impl<T, N> DataView for ProbabilitySet<T, N>
where
    T: DataView,
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
impl<T, N> DataViewMut for ProbabilitySet<T, N>
where
    T: DataViewMut,
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
impl<T, N> ContiguousDataView for ProbabilitySet<T, N> where T: ContiguousDataView {}
impl<T, N> SliceView for ProbabilitySet<T, N>
where
    T: SliceView,
{
    #[inline]
    fn slice(&self) -> &[Self::Value] { self.data.slice() }
}
impl<T, N> SliceViewMut for ProbabilitySet<T, N>
where
    T: SliceViewMut,
{
    #[inline]
    fn slice_mut(&mut self) -> &mut [Self::Value] { self.data.slice_mut() }
}
impl<T, N> ConstructableDataView for ProbabilitySet<T, N>
where
    T: ConstructableDataView,
{
    type ConstructableContainer<V> = T::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        T::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        T::try_from_iter(iter)
    }
}

impl<T, N> ProbabilitySet<T, N>
where
    T: DataView<Value = Probability<N>>,
    N: Number,
{
    /// Constructs a new set
    #[inline]
    pub fn new(data: T, ctx: ProbabilityContext<N>) -> Self { Self { data, ctx } }
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
    /// Constructs a new set from [`ProbabilitiesSpec`]
    /// # Panics
    /// Panics if `PO::Real` cannot be cast to `PO::Value`, or if any probability is incorrectly
    /// specified.
    #[inline]
    pub fn from_opts<PO>(opts: &PO, eps: Epsilon<PO::Real>) -> Self
    where
        PO: ProbabilitiesSpec<Value = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let eps_inner = <PO::Value as NumCast>::from(eps.get()).expect("eps converts to native");
        let ctx =
            ProbabilityContext::new(opts.max(), Epsilon::new(eps_inner).expect("eps in [0,1)"))
                .expect("max is finite");
        let data =
            opts.iter_map(|(i, v)| (i, Probability::new(*v, &ctx).expect("probability in [0,1]")));
        Self::new(data, ctx)
    }
    /// Constructs a new real-valued set from [`ProbabilitiesSpec`]
    /// # Panics
    /// Panics if any probability is incorrectly specified.
    #[inline]
    pub fn from_opts_real<PO>(opts: &PO, eps: Epsilon<PO::Real>) -> Self
    where
        N: NumberFloat,
        PO: ProbabilitiesSpec<Real = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let ctx = ProbabilityContext::new_real(eps);
        let data = PO::from_iter(opts.entries_real().map(|(i, v)| {
            (
                i,
                Probability::new(v, &ctx).expect("probability can be constructed"),
            )
        }));
        Self::new(data, ctx)
    }
}

impl<T, N> WeightCollection<T::Id> for ProbabilitySet<T, N>
where
    T: DataView<Value = Probability<N>>,
    N: Number,
{
    #[inline]
    fn get_weight(&self, id: T::Id) -> Option<f64> {
        self.get(id)
            .map(|v| v.get().to_f64().expect("convert to f64"))
    }
}

/// Probability store
pub trait ProbabilityStore: DataViewMut<Value = Probability<Self::N>> {
    /// The base probability type
    type N: Number;
    /// Returns a reference to the probability context
    fn ctx(&self) -> &ProbabilityContext<Self::N>;
    /// Returns `true` if the probability has a zero-value
    #[must_use]
    #[inline]
    fn is_zero(&self, id: Self::Id) -> Option<bool> { self.get(id).map(|v| v.is_zero(self.ctx())) }
    /// Returns `true` if the probability has neither a zero nor a full value
    #[must_use]
    #[inline]
    fn is_partial(&self, id: Self::Id) -> Option<bool> {
        self.get(id).map(|v| v.is_partial(self.ctx()))
    }
    /// Returns `true` if the probability has a full-value (one)
    #[must_use]
    #[inline]
    fn is_full(&self, id: Self::Id) -> Option<bool> { self.get(id).map(|v| v.is_full(self.ctx())) }
    /// Adds `value` to the probability of `unit`.
    /// Returns whatever could not be added to `value`.
    #[inline]
    fn add(&mut self, id: Self::Id, value: Self::Value) -> Option<Self::Value> {
        let ctx = *self.ctx();
        self.get_mut(id).map(|v| v.add(value, &ctx))
    }
    /// Subtracts `value` from the probability of `unit`.
    /// Returns whatever could not be subtracted from `value`.
    #[inline]
    fn subtract(&mut self, id: Self::Id, value: Self::Value) -> Option<Self::Value> {
        self.get_mut(id).map(|v| v.subtract(value))
    }
    /// Sets the probability of unit `id` to `value`.
    /// # Panics
    /// Panics if `value` is not a valid probability representation.
    #[inline]
    fn set(&mut self, id: Self::Id, value: Self::Value) -> Option<()> {
        self.get_mut(id).map(|v| *v = value)
    }
    /// Sets the probability of unit `idx` to the zero representation.
    #[inline]
    fn set_zero(&mut self, id: Self::Id) -> Option<()> {
        self.get_mut(id).map(|v| *v = Self::Value::zero())
    }
    /// Sets the probability of unit `idx` to the full representation.
    #[inline]
    fn set_full(&mut self, id: Self::Id) -> Option<()> {
        let ctx = *self.ctx();
        self.get_mut(id).map(|v| *v = Self::Value::full(&ctx))
    }
    /// Draws a random value from the probability representation
    #[inline]
    fn draw<R>(&self, rng: &mut R) -> Self::Value
    where
        R: Rand<Self::N>,
    {
        let r = rng.rand_to(self.ctx().max);
        Self::Value::new(r, self.ctx()).expect("r < max")
    }
    /// Draws a random value from the probability representation up to `max`.
    /// # Panics
    /// Panics if not `0 < max <= self.max`
    #[inline]
    fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::Value
    where
        R: Rand<Self::N>,
    {
        assert!(
            Self::N::ZERO < max && max <= self.ctx().max,
            "0 < max <= repr max"
        );
        let r = rng.rand_to(max);
        Self::Value::new(r, self.ctx()).expect("r < max")
    }
    /// Returns the weight of `other` on `main`.
    #[inline]
    #[must_use]
    fn weight(&self, main: Self::Id, other: Self::Id) -> Option<f64> {
        self.weight_to(*self.get(main)?, other)
    }
    /// Returns the weight of `other` on a probability `prob`.
    /// # Panics
    /// If the probability representations is not convertible to [`f64`]
    #[inline]
    #[must_use]
    fn weight_to(&self, prob: Self::Value, other: Self::Id) -> Option<f64> {
        let max = self.ctx().max.to_f64().expect("max converts to f64");
        let p1 = self
            .get(other)?
            .get()
            .to_f64()
            .expect("prob converts to f64");
        let w = if prob.is_full(self.ctx()) {
            max - p1
        } else if prob.is_zero(self.ctx()) {
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
/// Probability stores that can convert the probabilities to `Self::N`.
pub trait ProbabilityStoreToRaw: ProbabilityStore + ConstructableDataView {
    /// Returns the probabilties contained in the set, as their raw representations, in some store
    #[must_use]
    fn to_raw(&self) -> <Self as ConstructableDataView>::ConstructableContainer<Self::N>;
}

impl<T, N> ProbabilityStore for ProbabilitySet<T, N>
where
    T: DataViewMut<Value = Probability<N>>,
    N: Number,
{
    type N = N;
    #[inline]
    fn ctx(&self) -> &ProbabilityContext<N> { &self.ctx }
}
impl<T, N> ProbabilityStoreToRaw for ProbabilitySet<T, N>
where
    T: DataViewMut<Value = Probability<N>> + ConstructableDataView,
    N: Number,
{
    #[inline]
    fn to_raw(&self) -> <Self as ConstructableDataView>::ConstructableContainer<Self::N> {
        self.iter_map(|(i, v)| (i, v.get()))
    }
}
