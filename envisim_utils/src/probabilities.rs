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

use crate::kd_tree::searcher::WeightCollection;
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
    #[inline]
    pub fn new(probabilities: Vec<f64>, eps: f64) -> Self {
        Self {
            data: probabilities,
            eps,
        }
    }
    #[inline]
    pub fn new_equal(spec: ProbabilitySpecEqual, eps: f64) -> Self {
        let p = usize_to_f64(spec.sample_size()) / usize_to_f64(spec.population_size().get());
        Self::new(vec![p; spec.population_size().get()], eps)
    }
    #[inline]
    pub fn new_equal_f64(prob: f64, population_size: usize, eps: f64) -> Self {
        Self::new(vec![prob; population_size], eps)
    }
    #[inline]
    pub fn from_iter(probabilities: impl IntoIterator<Item = f64>, eps: f64) -> Self {
        Self::new(probabilities.into_iter().collect(), eps)
    }

    #[inline]
    pub fn is_prob(p: f64) -> bool { (0.0..=1.0).contains(&p) }
    #[inline]
    pub fn eps(&self) -> f64 { self.eps }
    #[inline]
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 { self.weight_to(self.data[idx0], idx1) }
    #[inline]
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
    #[inline]
    pub fn new(probabilities: Vec<usize>) -> Self {
        Self {
            data: probabilities,
        }
    }
    #[inline]
    pub fn new_equal(spec: ProbabilitySpecEqual) -> Self {
        Self::new(vec![spec.sample_size(); spec.population_size().get()])
    }
    #[inline]
    pub fn is_prob(&self, p: usize) -> bool { (0..=self.max()).contains(&p) }
}
impl FromIterator<usize> for ExactProbabilities {
    #[inline]
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

pub trait ProbabilityStore {
    type PR: num_traits::NumAssign + Copy + PartialOrd;
    // + Default

    fn data(&self) -> &[Self::PR];
    fn data_mut(&mut self) -> &mut [Self::PR];
    #[inline]
    fn len(&self) -> usize { self.data().len() }
    #[inline]
    fn is_empty(&self) -> bool { self.data().is_empty() }
    fn max(&self) -> Self::PR;

    #[inline]
    fn get(&self, idx: usize) -> Self::PR { self.data()[idx] }
    #[inline]
    fn set(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] = value; }
    fn set_zero(&mut self, idx: usize);
    fn set_max(&mut self, idx: usize);
    #[inline]
    fn add(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] += value; }

    fn is_zero(&self, idx: usize) -> bool;
    fn is_max(&self, idx: usize) -> bool;
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool;
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR;
}
impl ProbabilityStore for FloatProbabilities {
    type PR = f64;
    #[inline]
    fn data(&self) -> &[Self::PR] { &self.data }
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::PR] { &mut self.data }
    #[inline]
    fn max(&self) -> Self::PR { 1.0 }

    #[inline]
    fn set_zero(&mut self, idx: usize) { self.data[idx] = 0.0; }
    #[inline]
    fn set_max(&mut self, idx: usize) { self.data[idx] = 1.0; }

    #[inline]
    fn is_zero(&self, idx: usize) -> bool { self.data[idx] <= self.eps }
    #[inline]
    fn is_max(&self, idx: usize) -> bool { self.data[idx] >= 1.0 - self.eps }
    #[inline]
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool { (a - b).abs() <= self.eps }
    #[inline]
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR {
        rng.rf64_to(max).expect("max to be non-negative")
    }
}
impl ProbabilityStore for ExactProbabilities {
    type PR = usize;
    #[inline]
    fn data(&self) -> &[Self::PR] { &self.data }
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::PR] { &mut self.data }
    #[inline]
    fn max(&self) -> Self::PR { self.data.len() }

    #[inline]
    fn set_zero(&mut self, idx: usize) { self.data[idx] = 0; }
    #[inline]
    fn set_max(&mut self, idx: usize) { self.data[idx] = self.max(); }

    #[inline]
    fn is_zero(&self, idx: usize) -> bool { self.data[idx] == 0 }
    #[inline]
    fn is_max(&self, idx: usize) -> bool { self.data[idx] == self.max() }
    #[inline]
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool { a == b }
    #[inline]
    fn draw<G: RandomNumberGenerator>(&self, rng: &mut G, max: Self::PR) -> Self::PR {
        rng.rusize_to(max)
    }
}

impl WeightCollection for FloatProbabilities {
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self.get(id) }
    #[inline]
    fn try_get_weight(&self, id: usize) -> Option<f64> { self.data().get(id).copied() }
}
