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

use std::ops::{
    Add,
    AddAssign,
    Div,
    DivAssign,
    Mul,
    MulAssign,
    Sub,
    SubAssign,
};

use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
};
use crate::utils::usize_to_f64;

pub struct FloatProbabilities {
    data: Vec<f64>,
    eps: f64,
}
impl FloatProbabilities {
    pub fn new(probabilities: Vec<f64>, eps: f64) -> Self {
        Self {
            data: probabilities,
            eps,
        }
    }
    pub fn new_equal(spec: ProbabilitySpecEqual, eps: f64) -> Self {
        let p = usize_to_f64(spec.sample_size()) / usize_to_f64(spec.population_size());
        Self::new(vec![p; spec.population_size()], eps)
    }
    pub fn new_equal_f64(prob: f64, population_size: usize, eps: f64) -> Self {
        Self::new(vec![prob; population_size], eps)
    }
    pub fn from_iter(probabilities: impl IntoIterator<Item = f64>, eps: f64) -> Self {
        Self::new(probabilities.into_iter().collect(), eps)
    }

    pub fn is_prob(p: f64) -> bool { (0.0..=1.0).contains(&p) }
    pub fn eps(&self) -> f64 { self.eps }
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 { self.weight_to(self.data[idx0], idx1) }
    pub fn weight_to(&self, prob: f64, idx1: usize) -> f64 {
        if prob + self.data[idx1] <= self.max() {
            self.data[idx1] / (self.max() - prob)
        } else {
            (self.max() - self.data[idx1]) / prob
        }
    }
}

pub struct ExactProbabilities {
    data: Vec<usize>,
}
impl ExactProbabilities {
    pub fn new(probabilities: Vec<usize>) -> Self {
        Self {
            data: probabilities,
        }
    }
    pub fn new_equal(spec: ProbabilitySpecEqual) -> Self {
        Self::new(vec![spec.sample_size(); spec.population_size()])
    }
    pub fn is_prob(&self, p: usize) -> bool { (0..=self.max()).contains(&p) }
}
impl FromIterator<usize> for ExactProbabilities {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

pub trait ProbabilityStore {
    type PR: Copy
        + Default
        + PartialOrd
        + Add<Output = Self::PR>
        + AddAssign
        + Sub<Output = Self::PR>
        + SubAssign
        + Mul<Output = Self::PR>
        + MulAssign
        + Div<Output = Self::PR>
        + DivAssign;

    fn data(&self) -> &[Self::PR];
    fn data_mut(&mut self) -> &mut [Self::PR];
    fn len(&self) -> usize { self.data().len() }
    fn is_empty(&self) -> bool { self.data().is_empty() }
    fn max(&self) -> Self::PR;

    fn get(&self, idx: usize) -> Self::PR { self.data()[idx] }
    fn set(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] = value; }
    fn set_zero(&mut self, idx: usize);
    fn set_max(&mut self, idx: usize);
    fn add(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] += value; }

    fn is_zero(&self, idx: usize) -> bool;
    fn is_max(&self, idx: usize) -> bool;
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool;
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR;
}
impl ProbabilityStore for FloatProbabilities {
    type PR = f64;
    fn data(&self) -> &[Self::PR] { &self.data }
    fn data_mut(&mut self) -> &mut [Self::PR] { &mut self.data }
    fn max(&self) -> Self::PR { 1.0 }

    fn set_zero(&mut self, idx: usize) { self.data[idx] = 0.0; }
    fn set_max(&mut self, idx: usize) { self.data[idx] = 1.0; }

    fn is_zero(&self, idx: usize) -> bool { self.data[idx] <= self.eps }
    fn is_max(&self, idx: usize) -> bool { self.data[idx] >= 1.0 - self.eps }
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool { (a - b).abs() <= self.eps }
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR {
        rng.rf64_to(max).expect("max to be non-negative")
    }
}
impl ProbabilityStore for ExactProbabilities {
    type PR = usize;
    fn data(&self) -> &[Self::PR] { &self.data }
    fn data_mut(&mut self) -> &mut [Self::PR] { &mut self.data }
    fn max(&self) -> Self::PR { self.data.len() }

    fn set_zero(&mut self, idx: usize) { self.data[idx] = 0; }
    fn set_max(&mut self, idx: usize) { self.data[idx] = self.max(); }

    fn is_zero(&self, idx: usize) -> bool { self.data[idx] == 0 }
    fn is_max(&self, idx: usize) -> bool { self.data[idx] == self.max() }
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool { a == b }
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR {
        rng.rusize_to(max)
    }
}
