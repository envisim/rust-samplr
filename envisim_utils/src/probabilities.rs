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
use std::usize;

use num_traits::{
    ConstOne,
    ConstZero,
};

use crate::kd_tree::searcher::WeightCollection;
use crate::number_traits::{
    Number,
    NumberFloat,
};
use crate::random::FloatRng;
use crate::sampling_options::Epsilon;

pub trait ProbabilityValue {
    type N: Number;
}
pub trait RealProbabilityValue: ProbabilityValue
where
    Self::N: NumberFloat,
{
    fn is_real_probability(prob: Self::N) -> bool { (Self::N::ZERO..Self::N::ONE).contains(&prob) }
    fn from_real(prob: Self::N, eps: Epsilon<Self::N>) -> Option<Probability<Self::N>> {
        Probability::from_raw(prob, Self::N::ONE, eps)
    }
}

/// Decision result
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::exhaustive_enums,
    reason = "a unit can only exists in three decision states"
)]
pub enum Probability<N> {
    Zero(N),
    Partial(N),
    Full(N),
}
impl<N> Probability<N> {
    pub fn is_zero(self) -> bool { matches!(self, Probability::Zero(_)) }
    pub fn is_partial(self) -> bool { matches!(self, Probability::Partial(_)) }
    pub fn is_full(self) -> bool { matches!(self, Probability::Full(_)) }
    pub fn is_probability(prob: N, max: N) -> bool
    where
        N: Number,
    {
        (N::ZERO..=max).contains(&prob)
    }

    pub fn get(self) -> N
    where
        N: Copy,
    {
        match self {
            Probability::Zero(p) | Probability::Partial(p) | Probability::Full(p) => p,
        }
    }

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
    fn from_raw_unchecked(prob: N, max: N, eps: Epsilon<N>) -> Self
    where
        N: Number,
    {
        // Now not infinite or nan
        if !(N::ZERO..=max).contains(&prob) {
            panic!("invalid prob {prob} (max: {max})");
        } else if eps.is_zero(prob) {
            Probability::Zero(N::ZERO)
        } else if eps.is_zero(max - prob) {
            Probability::Full(max)
        } else {
            Probability::Partial(prob)
        }
    }
}

impl<N> PartialEq for Probability<N> {
    fn eq(&self, other: &Self) -> bool {
        use Probability::*;
        match (self, other) {
            (Zero(_), Zero(_)) | (Full(_), Full(_)) => true,
            (Partial(a), Partial(b)) => a == b,
            _ => false,
        }
    }
}

impl<N> Eq for Probability<N> {}
impl<N> PartialOrd for Probability<N>
where
    N: PartialOrd,
{
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
    fn from_raw<I>(probs: I, max: Self::N, eps: Epsilon<Self::N>) -> Self
    where
        Self: Sized,
        I: IntoIterator<Item = Self::N>,
        Self::N: Number;
    fn draw<R>(&self, rng: &mut R) -> Self::N
    where
        R: FloatRng;
    fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
    where
        R: FloatRng;
}
pub trait RealProbabilityCollection: ProbabilityCollection
where
    Self::N: NumberFloat,
{
}

#[derive(Debug, Clone)]
pub struct ProbabilitySet<N> {
    data: Box<[Probability<N>]>,
    max: N,
    eps: Epsilon<N>,
}
impl<N> ProbabilitySet<N> {
    pub fn data(&self) -> &[Probability<N>] { &self.data }
    pub fn eps(&self) -> Epsilon<N>
    where
        N: Copy,
    {
        self.eps
    }
    pub fn max(&self) -> N
    where
        N: Copy,
    {
        self.max
    }
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.data.len()).expect("set to be non-empty")
    }
    pub fn get(&self, idx: usize) -> Option<&Probability<N>> { self.data.get(idx) }
    pub fn set(&mut self, idx: usize, value: N)
    where
        N: Number,
    {
        self[idx] = Probability::from_raw_unchecked(value, self.max, self.eps);
    }
    pub fn set_zero(&mut self, idx: usize)
    where
        N: Number,
    {
        self[idx] = Probability::Zero(N::ZERO);
    }
    pub fn set_full(&mut self, idx: usize) { self[idx] = Probability::Full(self.max); }
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
                    self[unit] = Probability::from_raw_unchecked(sum, self.max, self.eps);
                    Probability::Zero(N::ZERO)
                } else {
                    self[unit] = Probability::Full(self.max);
                    Probability::from_raw_unchecked(sum - self.max, self.max, self.eps)
                }
            }
        }
    }
    pub fn weight(&self, main: usize, other: usize) -> f64 { self.weight_to(self[main], other) }
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
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self[id].get().to_f64().expect("convert to f64") }
    #[inline]
    fn try_get_weight(&self, id: usize) -> Option<f64> {
        self.get(id)
            .map(|v| v.get().to_f64().expect("convert to f64"))
    }
}
impl<N> Index<usize> for ProbabilitySet<N> {
    type Output = Probability<N>;
    fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}
impl<N> IndexMut<usize> for ProbabilitySet<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.data[index] }
}

// Macros for implementing probabilities for all primitive number types
macro_rules! prob_float {
    ($t:ty,$r:expr) => {
        impl ProbabilityValue for Probability<$t> {
            type N = $t;
        }
        impl RealProbabilityValue for Probability<$t> {}
        impl ProbabilitySet<$t> {
            pub fn new<I>(probs: I, eps: Epsilon<$t>) -> Self
            where
                I: IntoIterator<Item = $t>,
            {
                Self::from_raw(probs, 1.0, eps)
            }
        }
        impl ProbabilityCollection for ProbabilitySet<$t> {
            type N = $t;
            fn from_raw<I>(probs: I, _max: Self::N, eps: Epsilon<Self::N>) -> Self
            where
                I: IntoIterator<Item = Self::N>,
                Self::N: Number,
            {
                let max = 1.0;
                let data: Box<[Probability<Self::N>]> = probs
                    .into_iter()
                    .map(|p| Probability::from_raw_unchecked(p, max, eps))
                    .collect();
                Self { data, max, eps }
            }
            fn draw<R>(&self, rng: &mut R) -> Self::N
            where
                R: FloatRng,
            {
                rng.$r()
            }
            fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
            where
                R: FloatRng,
            {
                assert!(0.0 < max && max <= 1.0);
                rng.$r() * max
            }
        }
        impl RealProbabilityCollection for ProbabilitySet<$t> {}
    };
}
macro_rules! prob_int {
    ($t:ty,$r:expr) => {
        impl ProbabilityValue for Probability<$t> {
            type N = $t;
        }
        impl ProbabilitySet<$t> {
            pub fn new<I>(probs: I, max: $t) -> Self
            where
                I: IntoIterator<Item = $t>,
            {
                Self::from_raw(probs, max, Default::default())
            }
        }
        impl ProbabilityCollection for ProbabilitySet<$t> {
            type N = $t;
            fn from_raw<I>(probs: I, max: Self::N, _eps: Epsilon<Self::N>) -> Self
            where
                I: IntoIterator<Item = Self::N>,
                Self::N: Number,
            {
                assert!(0 < max);
                let eps = 0;
                let data: Box<[Probability<Self::N>]> = probs
                    .into_iter()
                    .map(|p| Probability::from_raw_unchecked(p, max, eps))
                    .collect();
                Self { data, max, eps }
            }
            fn draw<R>(&self, rng: &mut R) -> Self::N
            where
                R: FloatRng,
            {
                self.draw_partial(rng, self.max)
            }
            fn draw_partial<R>(&self, rng: &mut R, max: Self::N) -> Self::N
            where
                R: FloatRng,
            {
                assert!(0 < max && max <= self.max);
                rng.$r(max)
            }
        }
    };
}

prob_int!(usize, rusize_to);
prob_int!(u8, ru8_to);
prob_int!(u16, ru16_to);
prob_int!(u32, ru16_to);
prob_int!(u64, ru16_to);
prob_int!(u128, ru16_to);

prob_float!(f32, rf32);
prob_float!(f64, rf64);
