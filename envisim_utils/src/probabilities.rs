// Copyright (C) 2025 Wilmer Prentius.
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

use std::cmp::PartialOrd;
use std::ops::{
    Add,
    AddAssign,
    Div,
    DivAssign,
    Index,
    IndexMut,
    Mul,
    MulAssign,
    Rem,
    Sub,
    SubAssign,
};
use std::slice::{
    Iter,
    IterMut,
};

use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};

pub trait Probabilities: Index<usize> + IndexMut<usize> + Clone {
    type Prob: Sized
        + Add<Output = Self::Prob>
        + AddAssign
        + Sub<Output = Self::Prob>
        + SubAssign
        + Mul<Output = Self::Prob>
        + MulAssign
        + Div<Output = Self::Prob>
        + DivAssign
        + Rem<Output = Self::Prob>
        + PartialOrd
        + Copy
        + Default;
    fn zero(&self) -> Self::Prob;
    fn one(&self) -> Self::Prob;
    /// Creates a new probability container, without checking if the probabilities are valid
    fn with_values_unchecked(values: Vec<Self::Prob>, eps: f64) -> Self;
    /// Creates a new probability container
    fn with_value(size: usize, value: Self::Prob, eps: f64) -> Result<Self, SamplingOptionsError>;
    /// Returns the length of the container
    fn len(&self) -> usize { self.data().len() }
    /// Returns `true` if the container is empty
    fn is_empty(&self) -> bool { self.data().is_empty() }
    /// Returns the epsilon value
    fn eps(&self) -> f64;
    /// Sets epsilon value
    fn set_eps(&mut self, eps: f64) -> Result<(), SamplingOptionsError>;
    /// Returns a reference to the underlying list of probabilities.
    fn data(&self) -> &[Self::Prob];
    /// Returns a mutable reference to the underlying list of probabilities.
    fn data_mut(&mut self) -> &mut [Self::Prob];
    /// Returns an iterator over the probabilities
    fn iter(&self) -> Iter<'_, Self::Prob> { self.data().iter() }
    /// Returns a mutable iterator over the probabilities
    fn iter_mut(&mut self) -> IterMut<'_, Self::Prob> { self.data_mut().iter_mut() }
    /// Returns `true` if the probability is less than `epsilon`
    fn is_zero(&self, idx: usize) -> bool { self.data()[idx] == self.zero() }
    /// Returns `true` if the probability is larger than `1.0 - epsilon`
    fn is_one(&self, idx: usize) -> bool { self.data()[idx] == self.one() }
    fn set_zero(&mut self, idx: usize) { self.data_mut()[idx] = self.zero(); }
    fn set_one(&mut self, idx: usize) { self.data_mut()[idx] = self.one(); }
    fn set(&mut self, idx: usize, p: Self::Prob) -> Result<(), SamplingOptionsError> {
        if !(self.zero()..=self.one()).contains(&p) {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        self.data_mut()[idx] = p;
        Ok(())
    }
    fn add(&mut self, idx: usize, p: Self::Prob) { self.data_mut()[idx] += p; }
    fn get(&self, idx: usize) -> Self::Prob { self.data()[idx] }
    fn draw<R>(&self, rng: &mut R, max: Self::Prob) -> Self::Prob
    where
        R: RandomNumberGenerator;
}

/// Container for handling unequal inclusion probabilities.
#[derive(Clone, Debug)]
pub struct ProbabilitiesUnequal {
    /// An epsilon, used for comparison of floats
    eps: f64,
    probabilities: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct ProbabilitiesEqual {
    /// An epsilon, used for comparison of floats
    eps: f64,
    probabilities: Vec<usize>,
}

impl Probabilities for ProbabilitiesUnequal {
    type Prob = f64;
    fn zero(&self) -> Self::Prob { 0.0 }
    fn one(&self) -> Self::Prob { 1.0 }
    fn with_values_unchecked(values: Vec<Self::Prob>, eps: f64) -> Self {
        Self {
            eps,
            probabilities: values,
        }
    }
    fn with_value(size: usize, value: Self::Prob, eps: f64) -> Result<Self, SamplingOptionsError> {
        if !(0.0..=1.0).contains(&value) {
            return Err(SamplingOptionsError::InvalidProbability);
        }

        Ok(Self {
            eps,
            probabilities: vec![value; size],
        })
    }
    fn eps(&self) -> f64 { self.eps }
    fn set_eps(&mut self, eps: f64) -> Result<(), SamplingOptionsError> {
        if !(0.0..1.0).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(())
    }
    fn data(&self) -> &[Self::Prob] { &self.probabilities }
    fn data_mut(&mut self) -> &mut [Self::Prob] { &mut self.probabilities }
    fn is_zero(&self, idx: usize) -> bool { self[idx] <= self.eps }
    fn is_one(&self, idx: usize) -> bool { 1.0 - self.eps <= self[idx] }
    fn draw<R>(&self, rng: &mut R, max: Self::Prob) -> Self::Prob
    where
        R: RandomNumberGenerator,
    {
        rng.rf64_to(max).expect("max to be non-negative")
    }
}
impl Probabilities for ProbabilitiesEqual {
    type Prob = usize;
    fn zero(&self) -> Self::Prob { 0 }
    fn one(&self) -> Self::Prob { self.len() }
    fn with_values_unchecked(values: Vec<Self::Prob>, eps: f64) -> Self {
        Self {
            eps,
            probabilities: values,
        }
    }
    fn with_value(size: usize, value: Self::Prob, eps: f64) -> Result<Self, SamplingOptionsError> {
        if value > size {
            return Err(SamplingOptionsError::InvalidProbability);
        }

        Ok(Self {
            eps,
            probabilities: vec![value; size],
        })
    }
    fn eps(&self) -> f64 { self.eps }
    fn set_eps(&mut self, eps: f64) -> Result<(), SamplingOptionsError> {
        if !(0.0..1.0).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(())
    }
    fn data(&self) -> &[Self::Prob] { &self.probabilities }
    fn data_mut(&mut self) -> &mut [Self::Prob] { &mut self.probabilities }
    fn draw<R>(&self, rng: &mut R, max: Self::Prob) -> Self::Prob
    where
        R: RandomNumberGenerator,
    {
        rng.rusize_to(max)
    }
}

impl ProbabilitiesUnequal {
    pub fn new<P, S, B>(options: &SamplingOptions<P, S, B>) -> Self
    where
        P: Probabilities,
    {
        Self {
            eps: options.eps(),
            probabilities: options.probabilities().slice().into(),
        }
    }
    /// Calulates the weight that can be assigned to the unit `idx1` from `idx0`
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 { self.weight_to(self[idx0], idx1) }

    /// Calulates the weight that can be assigned to the unit `idx1` from `prob`
    pub fn weight_to(&self, prob: f64, idx1: usize) -> f64 {
        if prob + self[idx1] <= 1.0 {
            self[idx1] / (1.0 - prob)
        } else {
            (1.0 - self[idx1]) / prob
        }
    }
}

impl ProbabilitiesEqual {
    pub fn new<S, B>(options: &SamplingOptions<Self, S, B>) -> Self { options.into() }
}

impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesUnequal, S, B>> for ProbabilitiesUnequal {
    fn from(options: &SamplingOptions<'a, ProbabilitiesUnequal, S, B>) -> Self {
        Self {
            eps: options.eps(),
            probabilities: options.probabilities().slice().into(),
        }
    }
}
impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesEqual, S, B>> for ProbabilitiesUnequal {
    fn from(options: &SamplingOptions<'a, ProbabilitiesEqual, S, B>) -> Self {
        Self {
            eps: options.eps(),
            probabilities: options.probabilities().slice().into(),
        }
    }
}
impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesEqual, S, B>> for ProbabilitiesEqual {
    fn from(options: &SamplingOptions<'a, ProbabilitiesEqual, S, B>) -> Self {
        Self {
            eps: options.eps(),
            probabilities: options.probabilities().slice_equal(),
        }
    }
}

impl Index<usize> for ProbabilitiesUnequal {
    // type Output = f64;
    type Output = <Self as Probabilities>::Prob;
    fn index(&self, idx: usize) -> &Self::Output { &self.probabilities[idx] }
}
impl Index<usize> for ProbabilitiesEqual {
    // type Output = usize;
    type Output = <Self as Probabilities>::Prob;
    fn index(&self, idx: usize) -> &Self::Output { &self.probabilities[idx] }
}
impl IndexMut<usize> for ProbabilitiesUnequal {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output { &mut self.probabilities[idx] }
}
impl IndexMut<usize> for ProbabilitiesEqual {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output { &mut self.probabilities[idx] }
}
