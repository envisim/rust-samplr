use crate::error::InputError;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

pub struct Probabilities {
    pub eps: f64,
    probabilities: Vec<f64>,
}

impl Probabilities {
    #[inline]
    pub fn new(length: usize, value: f64) -> Result<Self, InputError> {
        if value.is_nan() || (0.0..=1.0).contains(&value) {
            return Err(InputError::InvalidProbability);
        }

        Ok(Self {
            eps: 0.0,
            probabilities: vec![value; length],
        })
    }

    #[inline]
    pub fn with_values(values: &[f64]) -> Result<Self, InputError> {
        Self::check(values)?;

        Ok(Self {
            eps: 0.0,
            probabilities: values.to_vec(),
        })
    }

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

    #[inline]
    pub fn check_eps(eps: f64) -> Result<(), InputError> {
        if eps.is_nan() || (0.0..1.0).contains(&eps) {
            return Err(InputError::InvalidEpsilon);
        }

        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.probabilities.is_empty()
    }

    #[inline]
    pub fn is_zero(&self, idx: usize) -> bool {
        self[idx] <= self.eps
    }

    #[inline]
    pub fn is_one(&self, idx: usize) -> bool {
        1.0 - self.eps <= self[idx]
    }

    #[inline]
    pub fn weight(&self, idx0: usize, idx1: usize) -> f64 {
        self.weight_to(self[idx0], idx1)
    }

    #[inline]
    pub fn weight_to(&self, prob: f64, idx1: usize) -> f64 {
        if prob + self[idx1] <= 1.0 {
            self[idx1] / (1.0 - prob)
        } else {
            (1.0 - self[idx1]) / prob
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<f64> {
        self.probabilities.iter()
    }

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
