// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

use crate::error::InputError;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

/// Container for handling inclusion probabilities.
pub struct Probabilities {
    /// An epsilon, used for comparison of floats
    pub eps: f64,
    probabilities: Vec<f64>,
}

impl Probabilities {
    /// Constructs a new probability container, filled with `value`
    /// Returns error if `value` is [`f64::NAN`] or outside the range `(0.0..=1.0)`.
    #[inline]
    pub fn new(length: usize, value: f64) -> Result<Self, InputError> {
        if value.is_nan() || !(0.0..=1.0).contains(&value) {
            return Err(InputError::InvalidProbability);
        }

        Ok(Self {
            eps: 0.0,
            probabilities: vec![value; length],
        })
    }

    /// Constructs a new probability container, filled with values from `values`.
    /// Returns error if any value is [`f64::NAN`] or outside the range `(0.0..=1.0)`.
    #[inline]
    pub fn with_values(values: &[f64]) -> Result<Self, InputError> {
        Self::check(values)?;

        Ok(Self {
            eps: 0.0,
            probabilities: values.to_vec(),
        })
    }

    /// Returns error if any value is [`f64::NAN`] or outside the range `(0.0..=1.0)`.
    #[inline]
    pub fn check(probabilities: &[f64]) -> Result<(), InputError> {
        if probabilities
            .iter()
            .any(|p| p.is_nan() || !(0.0..=1.0).contains(p))
        {
            return Err(InputError::InvalidProbability);
        }

        Ok(())
    }

    /// Returns error if the epsilon is outside the range (0.0..1.0)
    #[inline]
    pub fn check_eps(eps: f64) -> Result<(), InputError> {
        if eps.is_nan() || !(0.0..1.0).contains(&eps) {
            return Err(InputError::InvalidEpsilon);
        }

        Ok(())
    }

    /// Returns a reference to the underlying list of probabilities.
    #[inline]
    pub fn data(&self) -> &[f64] {
        &self.probabilities
    }

    /// Returns length of the container.
    #[inline]
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }
    /// Returns `true` if the container is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    /// Returns `true` if the probability is less than `epsilon`
    #[inline]
    pub fn is_zero(&self, idx: usize) -> bool {
        self[idx] <= self.eps
    }

    /// Returns `true` if the probability is larger than `1.0 - epsilon`
    #[inline]
    pub fn is_one(&self, idx: usize) -> bool {
        1.0 - self.eps <= self[idx]
    }

    /// Calulates the weight that can be assigned to the unit `idx1` from `idx0`
    #[inline]
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 {
        self.weight_to(self[idx0], idx1)
    }

    /// Calulates the weight that can be assigned to the unit `idx1` from `prob`
    #[inline]
    pub fn weight_to(&self, prob: f64, idx1: usize) -> f64 {
        if prob + self[idx1] <= 1.0 {
            self[idx1] / (1.0 - prob)
        } else {
            (1.0 - self[idx1]) / prob
        }
    }

    /// Returns an iterator over the probabilities
    #[inline]
    pub fn iter(&self) -> Iter<f64> {
        self.probabilities.iter()
    }

    /// Returns a mutable iterator over the probabilities
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<f64> {
        self.probabilities.iter_mut()
    }
}

impl Index<usize> for Probabilities {
    type Output = f64;

    #[inline]
    fn index(&self, idx: usize) -> &f64 {
        &self.probabilities[idx]
    }
}

impl IndexMut<usize> for Probabilities {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.probabilities[idx]
    }
}

impl std::fmt::Debug for Probabilities {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("Probabilities")
            .field("eps", &self.eps)
            .field("prob", &self.probabilities)
            .finish()
    }
}
