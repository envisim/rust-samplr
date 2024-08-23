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

use crate::SampleOptions;
use envisim_utils::{Indices, Probabilities, SamplingError};
use rand::Rng;

pub struct Sample(Vec<usize>);

impl Sample {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Sample(Vec::<usize>::with_capacity(capacity))
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[inline]
    pub fn add(&mut self, idx: usize) {
        self.0.push(idx);
    }

    #[inline]
    pub fn sort(&mut self) -> &mut Self {
        self.0.sort_unstable();
        self
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.to_vec()
    }

    #[inline]
    pub fn get(&self) -> &[usize] {
        &self.0
    }
}

pub struct Container<'a, R>
where
    R: Rng + ?Sized,
{
    rng: &'a mut R,
    probabilities: Probabilities,
    indices: Indices,
    sample: Sample,
}

impl<'a, R> Container<'a, R>
where
    R: Rng + ?Sized,
{
    #[inline]
    pub fn new(rng: &'a mut R, options: &SampleOptions) -> Result<Self, SamplingError> {
        let population_size = options.probabilities.len();

        let mut container = Container {
            rng,
            probabilities: unsafe {
                Probabilities::with_values_uncheked(options.probabilities, options.eps)
            },
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };

        for i in 0..population_size {
            container.decide_unit(i)?;
        }

        Ok(container)
    }
    #[inline]
    pub fn new_boxed(rng: &'a mut R, options: &SampleOptions) -> Result<Box<Self>, SamplingError> {
        Self::new(rng, options).map(Box::new)
    }

    #[inline]
    pub fn rng(&mut self) -> &mut R {
        self.rng
    }

    #[inline]
    pub fn probabilities(&self) -> &Probabilities {
        &self.probabilities
    }

    #[inline]
    pub fn probabilities_mut(&mut self) -> &mut Probabilities {
        &mut self.probabilities
    }

    #[inline]
    pub fn indices(&self) -> &Indices {
        &self.indices
    }

    #[inline]
    pub fn indices_mut(&mut self) -> &mut Indices {
        &mut self.indices
    }

    #[inline]
    pub fn indices_draw(&mut self) -> Option<&usize> {
        self.indices.draw(self.rng)
    }

    #[inline]
    pub fn sample(&self) -> &Sample {
        &self.sample
    }

    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample {
        &mut self.sample
    }

    #[inline]
    pub fn population_size(&self) -> usize {
        self.probabilities.len()
    }

    #[inline]
    pub fn decide_unit(&mut self, idx: usize) -> Result<Option<bool>, SamplingError> {
        if self.probabilities.is_zero(idx) {
            self.indices.remove(idx)?;
            return Ok(Some(false));
        } else if self.probabilities.is_one(idx) {
            self.indices.remove(idx)?;
            self.sample.add(idx);
            return Ok(Some(true));
        }

        Ok(None)
    }

    #[inline]
    pub fn update_last_unit(&mut self) -> Option<usize> {
        let id = *self.indices.last()?;

        self.probabilities[id] = if self.rng.gen::<f64>() < self.probabilities[id] {
            1.0
        } else {
            0.0
        };

        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;
    use envisim_utils::InputError;

    #[test]
    fn decide_unit() -> Result<(), InputError> {
        let mut rng = seeded_rng();
        let options = SampleOptions::new(&PROB_10_E)?;

        let mut c = Container::new(&mut rng, &options).unwrap();
        c.probabilities_mut()[0] = 1.0;
        c.probabilities_mut()[1] = 0.0;
        assert_eq!(c.decide_unit(0).unwrap(), Some(true));
        assert_eq!(c.decide_unit(1).unwrap(), Some(false));
        assert_eq!(c.decide_unit(2).unwrap(), None);

        Ok(())
    }
}
