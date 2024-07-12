use std::ops::{Index, IndexMut};

pub struct Probabilities {
    pub eps: f64,
    probabilities: Vec<f64>,
}

impl Probabilities {
    #[inline]
    pub fn new(length: usize, value: f64) -> Self {
        Probabilities {
            eps: 0.0,
            probabilities: vec![value; length],
        }
    }

    #[inline]
    pub fn with_values(values: &[f64]) -> Self {
        Probabilities {
            eps: 0.0,
            probabilities: values.to_vec(),
        }
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
        if self[idx0] + self[idx1] <= 1.0 {
            self[idx1] / (1.0 - self[idx0])
        } else {
            (1.0 - self[idx1]) / self[idx0]
        }
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
