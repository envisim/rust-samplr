use std::{
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

pub struct Probabilities {
    pub eps: f64,
    probabilities: Vec<f64>,
}

impl Probabilities {
    #[inline]
    pub fn new(length: usize, value: f64) -> Self {
        Self {
            eps: 0.0,
            probabilities: vec![value; length],
        }
    }

    #[inline]
    pub fn with_values(values: &[f64]) -> Self {
        assert!(Self::check(values));

        Self {
            eps: 0.0,
            probabilities: values.to_vec(),
        }
    }

    #[inline]
    pub fn check(probabilities: &[f64]) -> bool {
        probabilities.iter().all(|&p| 0.0 <= p && p <= 1.0)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.probabilities.len()
    }

    #[inline]
    pub fn is_zero(&self, idx: usize) -> bool {
        self[idx] <= self.eps
    }

    #[inline]
    pub fn is_one(&self, idx: usize) -> bool {
        self[idx] >= 1.0 - self.eps
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
