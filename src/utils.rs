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

use crate::error::SamplingError;
use crate::sample_options::SampleOptions;
use envisim_utils::{kd_tree::Node, random::RandomNumberGenerator, Indices, Probabilities};

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

pub struct SampleContainer<'a, R>
where
    R: RandomNumberGenerator + ?Sized,
{
    options: &'a SampleOptions<'a>,
    rng: &'a mut R,
    probabilities: Probabilities,
    indices: Indices,
    sample: Sample,
    tree: Option<Box<Node<'a>>>,
}

impl<'a, R> SampleContainer<'a, R>
where
    R: RandomNumberGenerator + ?Sized,
{
    #[inline]
    pub fn new(rng: &'a mut R, options: &'a SampleOptions<'a>) -> Result<Self, SamplingError> {
        options.check_base()?;
        let probs = options.probabilities();
        let population_size = probs.len();

        let mut container = SampleContainer {
            options,
            rng,
            probabilities: unsafe { Probabilities::with_values_uncheked(probs, options.eps()) },
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
            tree: None,
        };

        for i in 0..population_size {
            container.decide_unit(i)?;
        }

        if let Some(spreading) = options.spreading() {
            spreading.check(population_size)?;
            let mut units = container.indices().to_vec();
            container.reset_tree(&mut units)?;
        }

        Ok(container)
    }
    #[inline]
    pub fn new_with_tree(
        rng: &'a mut R,
        options: &'a SampleOptions<'a>,
    ) -> Result<Self, SamplingError> {
        options.check_spreading()?;
        SampleContainer::new(rng, options)
    }
    #[inline]
    pub fn reset_tree(&mut self, units: &mut [usize]) -> Result<&mut Self, SamplingError> {
        if let Some(spreading) = self.options.spreading() {
            self.tree = Some(spreading.build_tree(units)?);
        }

        Ok(self)
    }

    #[inline]
    pub fn options(&self) -> &'a SampleOptions<'a> {
        self.options
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
    pub fn sample(&self) -> &Sample {
        &self.sample
    }
    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample {
        &mut self.sample
    }
    #[inline]
    pub fn tree(&self) -> Option<&Node<'a>> {
        self.tree.as_deref()
    }
    #[inline]
    pub fn tree_mut(&mut self) -> Option<&mut Node<'a>> {
        self.tree.as_deref_mut()
    }

    #[inline]
    pub fn set_probability_and_decide(
        &mut self,
        idx: usize,
        prob: f64,
    ) -> Result<Option<bool>, SamplingError> {
        self.probabilities[idx] = prob;
        self.decide_unit(idx)
    }
    #[inline]
    pub fn add_probability_and_decide(
        &mut self,
        idx: usize,
        prob: f64,
    ) -> Result<Option<bool>, SamplingError> {
        self.probabilities[idx] += prob;
        self.decide_unit(idx)
    }
    #[inline]
    pub fn indices_draw(&mut self) -> Option<&usize> {
        self.indices.draw(self.rng)
    }
    #[inline]
    pub fn population_size(&self) -> usize {
        self.probabilities.len()
    }

    #[inline]
    pub fn decide_unit(&mut self, idx: usize) -> Result<Option<bool>, SamplingError> {
        let mut is_one = false;

        if self.probabilities.is_one(idx) {
            self.sample.add(idx);
            is_one = true;
        } else if !self.probabilities.is_zero(idx) {
            return Ok(None);
        }

        self.indices.remove(idx)?;

        if let Some(tree) = self.tree.as_mut() {
            tree.remove_unit(idx)?;
        }

        Ok(Some(is_one))
    }

    #[inline]
    pub fn update_last_unit(&mut self) -> Result<Option<bool>, SamplingError> {
        let &id = match self.indices.last() {
            Some(v) => v,
            None => return Ok(None),
        };

        if self.rng.rbern(self.probabilities[id]).unwrap() {
            self.set_probability_and_decide(id, 1.0)
        } else {
            self.set_probability_and_decide(id, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;
    use envisim_utils::{random::*, InputError};

    #[test]
    fn decide_unit() -> Result<(), InputError> {
        let mut rng = SmallRng::seed_from_u64(42);
        let options = SampleOptions::new(&PROB_10_E)?;

        let mut c = SampleContainer::new(&mut rng, &options).unwrap();
        c.probabilities_mut()[0] = 1.0;
        c.probabilities_mut()[1] = 0.0;
        assert_eq!(c.decide_unit(0).unwrap(), Some(true));
        assert_eq!(c.decide_unit(1).unwrap(), Some(false));
        assert_eq!(c.decide_unit(2).unwrap(), None);

        Ok(())
    }
}
