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

use num_traits::ToPrimitive;

use crate::kd_tree::searcher::WeightCollection;
use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
};

/// Decision result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "a unit can only exists in three decision states"
)]
pub enum UnitDecisionStatus {
    In,
    Out,
    Undecided,
}

/// Contains probabilities represented as floats
#[must_use]
#[derive(Debug, Clone)]
pub struct FloatProbabilities {
    /// Internal data
    data: Vec<f64>,
    /// An epsilon to be used when comparing probabilities
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
    #[expect(
        clippy::missing_panics_doc,
        clippy::unwrap_used,
        reason = "usize to f64 conversion"
    )]
    #[inline]
    pub fn new_equal(spec: ProbabilitySpecEqual, eps: f64) -> Self {
        let p =
            spec.sample_size().to_f64().unwrap() / spec.population_size().get().to_f64().unwrap();
        Self::new(vec![p; spec.population_size().get()], eps)
    }
    #[inline]
    pub fn new_equal_f64(prob: f64, population_size: usize, eps: f64) -> Self {
        Self::new(vec![prob; population_size], eps)
    }
    #[inline]
    pub fn from_iter<I>(probabilities: I, eps: f64) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        Self::new(probabilities.into_iter().collect(), eps)
    }

    #[must_use]
    #[inline]
    pub fn is_prob(p: f64) -> bool { (0.0..=1.0).contains(&p) }
    #[must_use]
    #[inline]
    pub fn eps(&self) -> f64 { self.eps }
    #[must_use]
    #[inline]
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 { self.weight_to(self.data[idx0], idx1) }
    #[must_use]
    #[inline]
    pub fn weight_to(&self, prob: f64, idx1: usize) -> f64 {
        if prob + self.data[idx1] <= self.max() {
            self.data[idx1] / (self.max() - prob)
        } else {
            (self.max() - self.data[idx1]) / prob
        }
    }
}

/// Contains probabilities represented as integers
#[must_use]
#[derive(Debug, Clone)]
pub struct ExactProbabilities {
    /// Internal data
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
    #[must_use]
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

    #[must_use]
    fn data(&self) -> &[Self::PR];
    #[must_use]
    fn data_mut(&mut self) -> &mut [Self::PR];
    #[must_use]
    #[inline]
    fn len(&self) -> usize { self.data().len() }
    #[must_use]
    #[inline]
    fn is_empty(&self) -> bool { self.data().is_empty() }
    fn max(&self) -> Self::PR;

    #[must_use]
    #[inline]
    fn get(&self, idx: usize) -> Self::PR { self.data()[idx] }
    #[inline]
    fn set(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] = value; }
    fn set_zero(&mut self, idx: usize);
    fn set_max(&mut self, idx: usize);
    #[inline]
    fn add(&mut self, idx: usize, value: Self::PR) { self.data_mut()[idx] += value; }

    fn unit_status(&self, idx: usize) -> UnitDecisionStatus;
    #[must_use]
    #[inline]
    fn is_zero(&self, idx: usize) -> bool { self.unit_status(idx) == UnitDecisionStatus::Out }
    #[must_use]
    #[inline]
    fn is_max(&self, idx: usize) -> bool { self.unit_status(idx) == UnitDecisionStatus::In }
    #[must_use]
    fn eq(&self, a: Self::PR, b: Self::PR) -> bool;
    #[must_use]
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
    fn unit_status(&self, idx: usize) -> UnitDecisionStatus {
        if self.data[idx] <= self.eps {
            UnitDecisionStatus::Out
        } else if self.data[idx] >= 1.0 - self.eps {
            UnitDecisionStatus::In
        } else {
            UnitDecisionStatus::Undecided
        }
    }
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
    fn unit_status(&self, idx: usize) -> UnitDecisionStatus {
        if self.data[idx] == 0 {
            UnitDecisionStatus::Out
        } else if self.data[idx] == self.max() {
            UnitDecisionStatus::In
        } else {
            UnitDecisionStatus::Undecided
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    static PROBABILITY_ARR: [f64; 6] = [0.1, 0.2, 0.0, 1.0, 0.6, 0.8];
    fn prob_new(eps: f64) -> FloatProbabilities {
        FloatProbabilities::new(PROBABILITY_ARR.to_vec(), eps)
    }

    #[test]
    fn is_zero() {
        let p = prob_new(f64::TEST_EPS);
        assert!(!p.is_zero(0));
        assert!(p.is_zero(2));
        assert!(!p.is_max(0));
        assert!(p.is_max(3));

        let mut p = prob_new(1e-2);
        p.set(0, 0.999);
        assert!(p.is_max(0));
    }

    #[test]
    fn weight() {
        let p = prob_new(f64::TEST_EPS);
        assert_delta!(p.weight(0, 1), 0.2 / 0.9);
        assert_delta!(p.weight_to(0.1, 1), 0.2 / 0.9);
        assert_delta!(p.weight(4, 5), 0.2 / 0.6);
        assert_delta!(p.weight_to(0.6, 5), 0.2 / 0.6);
    }
}
