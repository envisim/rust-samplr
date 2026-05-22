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

use std::cmp::Ordering;
use std::num::NonZeroUsize;
use std::ops::{
    Index,
    IndexMut,
};

use num_traits::{
    ConstOne,
    ConstZero,
};

use crate::kd_tree::searcher::WeightCollection;
use crate::number_traits::{
    Number,
    NumberFloat,
};
use crate::random::Rand;
use crate::sampling_options::Epsilon;

pub trait ProbabilityValue {
    type N: Number;
}
pub trait RealProbabilityValue: ProbabilityValue
where
    Self::N: NumberFloat,
{
    /// Returns `true` if `prob` is contained in 0.0..=1.0
    #[must_use]
    #[inline]
    fn is_real_probability(prob: Self::N) -> bool { (Self::N::ZERO..=Self::N::ONE).contains(&prob) }
    /// Constructs a [`Probability`] if `prob` can be converted, otherwise returns `None`.
    #[inline]
    fn from_real(prob: Self::N, eps: Epsilon<Self::N>) -> Option<Probability<Self::N>> {
        Probability::from_raw(prob, Self::N::ONE, eps)
    }
}

/// Stores a probability representation
#[expect(
    clippy::exhaustive_enums,
    reason = "a unit can only exists in three decision states"
)]
#[must_use]
#[derive(Debug, Clone, Copy)]
pub enum Probability<N> {
    Zero(N),
    Partial(N),
    Full(N),
}
impl<N> Probability<N> {
    /// Returns `true` if the probability has a zero-value
    #[must_use]
    #[inline]
    pub fn is_zero(self) -> bool { matches!(self, Probability::Zero(_)) }
    /// Returns `true` if the probability has neither a zero nor a full value
    #[must_use]
    #[inline]
    pub fn is_partial(self) -> bool { matches!(self, Probability::Partial(_)) }
    /// Returns `true` if the probability has a full-value (one)
    #[must_use]
    #[inline]
    pub fn is_full(self) -> bool { matches!(self, Probability::Full(_)) }
    /// Returns `true` if the candidate `prob` is a probability.
    #[must_use]
    #[inline]
    pub fn is_probability(prob: N, max: N) -> bool
    where
        N: Number,
    {
        (N::ZERO..=max).contains(&prob)
    }
    /// Returns the internal value of the probability
    #[must_use]
    #[inline]
    pub fn get(self) -> N
    where
        N: Copy,
    {
        match self {
            Probability::Zero(p) | Probability::Partial(p) | Probability::Full(p) => p,
        }
    }
    /// Constructs a probability from `prob` if it can be contained `0..=max`, otherwise returns
    /// `None`.
    #[inline]
    pub fn from_raw(prob: N, max: N, eps: Epsilon<N>) -> Option<Self>
    where
        N: Number,
    {
        // Now not infinite or nan
        if !(N::ZERO..=max).contains(&prob) {
            None
        } else if eps.is_zero(prob) {
            Some(Probability::Zero(N::ZERO))
        } else if eps.is_zero(max - prob) {
            Some(Probability::Full(max))
        } else {
            Some(Probability::Partial(prob))
        }
    }
}

impl<N> PartialEq for Probability<N>
where
    N: PartialEq,
{
    #[expect(clippy::enum_glob_use, reason = "small function")]
    #[must_use]
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        use Probability::*;
        match (self, other) {
            (Zero(_), Zero(_)) | (Full(_), Full(_)) => true,
            (Partial(a), Partial(b)) => a == b,
            _ => false,
        }
    }
}

impl<N> Eq for Probability<N> where N: PartialEq {}
impl<N> PartialOrd for Probability<N>
where
    N: PartialOrd,
{
    #[expect(clippy::enum_glob_use, reason = "small function")]
    #[must_use]
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use Probability::*;
        match (self, other) {
            (Zero(_), Zero(_)) | (Full(_), Full(_)) => Some(Ordering::Equal),
            (Zero(_), _) | (_, Full(_)) => Some(Ordering::Less),
            (Full(_), _) | (_, Zero(_)) => Some(Ordering::Greater),
            (Partial(a), Partial(b)) => a.partial_cmp(b),
        }
    }
}

pub trait ProbabilityCollection {
    type N: Number;
    #[must_use]
    fn from_raw<I>(probs: I, max: Self::N, eps: Epsilon<Self::N>) -> Self
    where
        Self: Sized,
        I: IntoIterator<Item = Self::N>,
        Self::N: Number;
    #[must_use]
    fn draw<R>(&self, rng: &mut R) -> Self::N
    where
        R: Rand<Self::N>;
    #[must_use]
    fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
    where
        R: Rand<Self::N>;
}
pub trait RealProbabilityCollection: ProbabilityCollection
where
    Self::N: NumberFloat,
{
}

#[must_use]
#[derive(Debug, Clone)]
pub struct ProbabilitySet<N> {
    /// The internal storage for the probability representations
    data: Box<[Probability<N>]>,
    /// The maximum value of the probability representations
    max: N,
    /// The epsilon value for comparison between two probability representations
    eps: Epsilon<N>,
}
impl<N> ProbabilitySet<N> {
    /// Returns the slice of the probabilties contained in the set
    #[inline]
    pub fn data(&self) -> &[Probability<N>] { &self.data }
    /// Returns the stored epsilon value
    #[inline]
    pub fn eps(&self) -> Epsilon<N>
    where
        N: Copy,
    {
        self.eps
    }
    /// Returns the maximum value of the probability representation
    #[must_use]
    #[inline]
    pub fn max(&self) -> N
    where
        N: Copy,
    {
        self.max
    }
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
    /// Returns the size of the set
    #[must_use]
    #[inline]
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.data.len()).expect("set to be non-empty")
    }
    /// Returns a reference to the probability for unit `idx`, or `None` if it does not exist.
    #[must_use]
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&Probability<N>> { self.data.get(idx) }
    /// Sets the probability of unit `idx` to `value`.
    /// # Panics
    /// Panics if `value` is not a valid probability representation.
    #[inline]
    pub fn set(&mut self, idx: usize, value: N)
    where
        N: Number,
    {
        self[idx] = Probability::from_raw(value, self.max, self.eps)
            .expect("value to be contained in 0..=max");
    }
    /// Sets the probability of unit `idx` to the zero representation.
    #[inline]
    pub fn set_zero(&mut self, idx: usize)
    where
        N: Number,
    {
        self[idx] = Probability::Zero(N::ZERO);
    }
    /// Sets the probability of unit `idx` to the full representation.
    #[inline]
    pub fn set_full(&mut self, idx: usize)
    where
        N: Copy,
    {
        self[idx] = Probability::Full(self.max);
    }
    /// Adds `value` to the probability of `unit`.
    /// Returns whatever could not be added to `value`.
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
    #[inline]
    pub fn add(&mut self, unit: usize, value: Probability<N>) -> Probability<N>
    where
        N: Number,
    {
        match (self[unit], value) {
            (Probability::Zero(_), _) | (Probability::Partial(_), Probability::Full(_)) => {
                let current = self[unit];
                self[unit] = value;
                current
            }
            (Probability::Full(_), _) | (Probability::Partial(_), Probability::Zero(_)) => value,
            (Probability::Partial(org), Probability::Partial(val)) => {
                let sum = org + val;
                if sum < self.max {
                    self[unit] = Probability::from_raw(sum, self.max, self.eps)
                        .expect("sum to be contained in 0..=max");
                    Probability::Zero(N::ZERO)
                } else {
                    self[unit] = Probability::Full(self.max);
                    Probability::from_raw(sum - self.max, self.max, self.eps)
                        .expect("sum-max to be contained in 0..=max")
                }
            }
        }
    }
    /// Returns the weight of `other` on `main`.
    #[inline]
    #[must_use]
    pub fn weight(&self, main: usize, other: usize) -> f64
    where
        N: Number,
    {
        self.weight_to(self[main], other)
    }
    /// Returns the weight of `other` on a probability `prob`.
    /// # Panics
    /// If the probability representations is not convertible to [`f64`]
    #[inline]
    #[must_use]
    pub fn weight_to(&self, prob: Probability<N>, other: usize) -> f64
    where
        N: Number,
    {
        let max = self.max.to_f64().expect("max to convert to f64");
        let p1 = self[other].get().to_f64().expect("to convert to f64");
        match prob {
            Probability::Partial(p0) => {
                let p0 = p0.to_f64().expect("to convert to f64");
                if p0 + p1 <= max {
                    p1 / (max - p0)
                } else {
                    (max - p1) / p0
                }
            }
            Probability::Zero(_) => p1,
            Probability::Full(_) => max - p1,
        }
    }
}

impl<N> WeightCollection for ProbabilitySet<N>
where
    N: Number,
{
    /// Returns the weight of unit `id`
    #[must_use]
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self[id].get().to_f64().expect("convert to f64") }
    /// Returns the weight of unit `id`, or `None` i the unit does not exist.
    #[must_use]
    #[inline]
    fn try_get_weight(&self, id: usize) -> Option<f64> {
        self.get(id)
            .map(|v| v.get().to_f64().expect("convert to f64"))
    }
}
impl<N> Index<usize> for ProbabilitySet<N> {
    type Output = Probability<N>;
    #[must_use]
    #[inline]
    fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}
impl<N> IndexMut<usize> for ProbabilitySet<N> {
    #[must_use]
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.data[index] }
}

/// Implements probability representations and sets for floats
macro_rules! prob_repr_impl_float {
    ($t:ty) => {
        impl ProbabilityValue for Probability<$t> {
            type N = $t;
        }
        impl RealProbabilityValue for Probability<$t> {}
        impl ProbabilitySet<$t> {
            /// Constructs a new probability set
            #[inline]
            pub fn new<I>(probs: I, eps: Epsilon<$t>) -> Self
            where
                I: IntoIterator<Item = $t>,
            {
                Self::from_raw(probs, 1.0, eps)
            }
        }
        impl ProbabilityCollection for ProbabilitySet<$t> {
            type N = $t;
            #[inline]
            fn from_raw<I>(probs: I, _max: Self::N, eps: Epsilon<Self::N>) -> Self
            where
                I: IntoIterator<Item = Self::N>,
                Self::N: Number,
            {
                let max = 1.0;
                let data: Box<[Probability<Self::N>]> = probs
                    .into_iter()
                    .map(|p| {
                        Probability::from_raw(p, max, eps).expect("p to be contained in 0..=max")
                    })
                    .collect();
                Self { data, max, eps }
            }
            #[inline]
            fn draw<R>(&self, rng: &mut R) -> Self::N
            where
                R: Rand<$t>,
            {
                rng.rand()
            }
            #[inline]
            fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
            where
                R: Rand<$t>,
            {
                assert!(0.0 < max && max <= 1.0);
                rng.rand_to(max)
            }
        }
        impl RealProbabilityCollection for ProbabilitySet<$t> {}
    };
}
/// Implements probability representations and sets for ints
macro_rules! prob_repr_impl_int {
    ($t:ty) => {
        impl ProbabilityValue for Probability<$t> {
            type N = $t;
        }
        impl ProbabilitySet<$t> {
            /// Constructs a new probability set
            #[inline]
            pub fn new<I>(probs: I, max: $t) -> Self
            where
                I: IntoIterator<Item = $t>,
            {
                Self::from_raw(probs, max, Default::default())
            }
        }
        impl ProbabilityCollection for ProbabilitySet<$t> {
            type N = $t;
            #[inline]
            fn from_raw<I>(probs: I, max: Self::N, _eps: Epsilon<Self::N>) -> Self
            where
                I: IntoIterator<Item = Self::N>,
                Self::N: Number,
            {
                assert!(0 < max);
                let eps = Epsilon::<$t>::default();
                let data: Box<[Probability<Self::N>]> = probs
                    .into_iter()
                    .map(|p| {
                        Probability::from_raw(p, max, eps).expect("p to be contained in 0..=max")
                    })
                    .collect();
                Self { data, max, eps }
            }
            #[inline]
            fn draw<R>(&self, rng: &mut R) -> Self::N
            where
                R: Rand<$t>,
            {
                rng.rand_to(self.max)
            }
            #[inline]
            fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
            where
                R: Rand<$t>,
            {
                assert!(0 < max && max <= self.max);
                rng.rand_to(max)
            }
        }
    };
}

prob_repr_impl_int!(usize);
prob_repr_impl_int!(u8);
prob_repr_impl_int!(u16);
prob_repr_impl_int!(u32);
prob_repr_impl_int!(u64);
prob_repr_impl_int!(u128);

prob_repr_impl_float!(f64);
