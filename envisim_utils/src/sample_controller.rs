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

use crate::indices::Indices;
use crate::matrix::MatrixTree;
use crate::probabilities::ProbabilityStore;
use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    SamplingOptionsResult,
    SpreadingOptions,
};

/// Sample container
pub struct Sample(Vec<usize>);
impl Sample {
    pub fn new(capacity: usize) -> Self { Sample(Vec::<usize>::with_capacity(capacity)) }
    pub fn clear(&mut self) { self.0.clear(); }
    pub fn add(&mut self, idx: usize) { self.0.push(idx); }
    pub fn sort(&mut self) -> &mut Self {
        self.0.sort_unstable();
        self
    }
    pub fn to_vec(&self) -> Vec<usize> { self.0.to_vec() }
    pub fn sort_to_vec(&mut self) -> Vec<usize> { self.sort().to_vec() }
    pub fn get(&self) -> &[usize] { &self.0 }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

/// Decision result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::exhaustive_enums)]
pub enum DecideUnit {
    In,
    Out,
    Undecided,
}

pub trait SampleController {
    type Store: ProbabilityStore;
    fn controller(&self) -> &BasicSampleController<Self::Store>;
    fn controller_mut(&mut self) -> &mut BasicSampleController<Self::Store>;
    fn probabilities(&self) -> &Self::Store { &self.controller().probabilities }
    fn probabilities_mut(&mut self) -> &mut Self::Store { &mut self.controller_mut().probabilities }
    fn indices(&self) -> &Indices { &self.controller().indices }
    fn indices_mut(&mut self) -> &mut Indices { &mut self.controller_mut().indices }
    fn sample(&self) -> &Sample { &self.controller().sample }
    fn sample_mut(&mut self) -> &mut Sample { &mut self.controller_mut().sample }

    fn population_size(&self) -> usize { self.probabilities().len() }

    fn draw<R: RandomNumberGenerator>(
        &self,
        rng: &mut R,
        max: <Self::Store as ProbabilityStore>::PR,
    ) -> <Self::Store as ProbabilityStore>::PR {
        self.probabilities().draw(rng, max)
    }

    fn unit_remove(&mut self, idx: usize) -> Option<usize> { self.indices_mut().remove(idx) }
    fn unit_decide(&mut self, idx: usize) -> Option<DecideUnit> {
        if self.probabilities().is_max(idx) {
            self.sample_mut().add(idx);
            self.unit_remove(idx)?;
            return Some(DecideUnit::In);
        } else if self.probabilities().is_zero(idx) {
            self.unit_remove(idx)?;
            return Some(DecideUnit::Out);
        }
        Some(DecideUnit::Undecided)
    }
    fn unit_set_and_decide(
        &mut self,
        idx: usize,
        prob: <Self::Store as ProbabilityStore>::PR,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().set(idx, prob);
        self.unit_decide(idx)
    }
    fn unit_set_max(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_max(idx);
        self.sample_mut().add(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::In)
    }
    fn unit_set_zero(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_zero(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::Out)
    }
    fn unit_add_and_decide(
        &mut self,
        idx: usize,
        prob: <Self::Store as ProbabilityStore>::PR,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().add(idx, prob);
        self.unit_decide(idx)
    }
    fn unit_decide_last<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Option<DecideUnit> {
        let Some(id) = self.indices().last() else {
            return Some(DecideUnit::Undecided);
        };
        let prob = self.probabilities().get(id);
        let max = self.probabilities().max();

        if self.draw(rng, max) < prob {
            self.unit_set_max(id)
        } else {
            self.unit_set_zero(id)
        }
    }
}

// BasicSampleController
pub struct BasicSampleController<ST: ProbabilityStore> {
    probabilities: ST,
    indices: Indices,
    sample: Sample,
}
impl<ST: ProbabilityStore> BasicSampleController<ST> {
    pub fn new(probabilities: ST) -> Self {
        let population_size = probabilities.len();
        let mut controller = Self {
            probabilities,
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };
        controller.init();
        controller
    }
    fn init(&mut self) {
        for i in 0..self.population_size() {
            self.unit_decide(i);
        }
    }
}

impl<ST: ProbabilityStore> SampleController for BasicSampleController<ST> {
    type Store = ST;
    fn controller(&self) -> &BasicSampleController<ST> { self }
    fn controller_mut(&mut self) -> &mut BasicSampleController<ST> { self }
    fn probabilities(&self) -> &ST { &self.probabilities }
    fn probabilities_mut(&mut self) -> &mut ST { &mut self.probabilities }
    fn indices(&self) -> &Indices { &self.indices }
    fn indices_mut(&mut self) -> &mut Indices { &mut self.indices }
    fn sample(&self) -> &Sample { &self.sample }
    fn sample_mut(&mut self) -> &mut Sample { &mut self.sample }
}

// SpreadingSampleController
pub struct SpreadingSampleController<'a, ST: ProbabilityStore> {
    controller: BasicSampleController<ST>,
    tree: MatrixTree<'a>,
}

impl<'a, ST: ProbabilityStore> SpreadingSampleController<'a, ST> {
    pub fn new(
        controller: BasicSampleController<ST>,
        spreading: &'a SpreadingOptions<'a>,
    ) -> SamplingOptionsResult<Self> {
        let mut units = controller.indices().to_vec();
        let tree = MatrixTree::new(spreading, &mut units);
        Ok(Self { controller, tree })
    }
    pub fn tree(&self) -> &MatrixTree<'a> { &self.tree }
    pub fn tree_mut(&mut self) -> &mut MatrixTree<'a> { &mut self.tree }
    pub fn reset_tree(
        &mut self,
        spreading: &'a SpreadingOptions<'a>,
        units: &mut [usize],
    ) -> SamplingOptionsResult<()> {
        self.tree = MatrixTree::new(spreading, units);
        Ok(())
    }
}

impl<'a, ST: ProbabilityStore> SampleController for SpreadingSampleController<'a, ST> {
    type Store = ST;
    fn controller(&self) -> &BasicSampleController<ST> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<ST> { &mut self.controller }

    fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        self.tree.remove_unit(idx)?;
        self.controller.unit_remove(idx)
    }
}

#[cfg(test)]
mod tests {
    // use envisim_test_utils::*;

    use super::*;
    use crate::sampling_options::{
        ProbabilitySpecUnequal,
        SamplingOptions,
    };

    #[test]
    fn basic_controller_from_options() {
        let probs = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
        let opts: SamplingOptions<ProbabilitySpecUnequal> = probs.try_into().unwrap();
        let controller = opts.to_controller();
        assert_eq!(controller.population_size(), 10);
    }

    #[test]
    fn basic_controller_exact() {
        let opts = SamplingOptions::new_equal(10, 3).unwrap();
        let controller = opts.to_controller();
        assert_eq!(controller.population_size(), 10);
        assert_eq!(controller.probabilities().max(), 10);
    }
}
