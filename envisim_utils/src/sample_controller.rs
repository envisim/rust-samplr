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

use std::num::NonZeroUsize;

use crate::indices::Indices;
use crate::kd_tree::{
    PointSet,
    Tree,
};
use crate::probabilities::{
    ProbabilityStore,
    UnitDecisionStatus,
};
use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    SamplingOptionsError,
    SamplingOptionsResult,
    SpreadingOptions,
};

/// Sample container
#[must_use]
#[derive(Debug, Clone)]
pub struct Sample(Vec<usize>);
impl Sample {
    #[inline]
    pub fn new(capacity: usize) -> Self { Sample(Vec::<usize>::with_capacity(capacity)) }
    #[inline]
    pub fn clear(&mut self) { self.0.clear(); }
    #[inline]
    pub fn add(&mut self, idx: usize) { self.0.push(idx); }
    #[inline]
    pub fn sort(&mut self) { self.0.sort_unstable(); }
    #[must_use]
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> { self.0.clone() }
    #[must_use]
    #[inline]
    pub fn sort_to_vec(&mut self) -> Vec<usize> {
        self.sort();
        self.to_vec()
    }
    #[must_use]
    #[inline]
    pub fn get(&self) -> &[usize] { &self.0 }
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize { self.0.len() }
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

#[must_use]
#[derive(Debug, Clone)]
pub struct SampleController<PROB, TREE = ()> {
    /// The probability store
    probabilities: PROB,
    /// The (remaining) sample indices
    indices: Indices,
    /// The units included in the sample
    sample: Sample,
    /// The kd-tree containing the (remaining) units
    tree: TREE,
}

impl<PROB, TREE> SampleController<PROB, TREE> {
    #[must_use]
    #[inline]
    pub fn probabilities(&self) -> &PROB { &self.probabilities }
    #[must_use]
    #[inline]
    pub fn probabilities_mut(&mut self) -> &mut PROB { &mut self.probabilities }
    #[inline]
    pub fn indices(&self) -> &Indices { &self.indices }
    #[inline]
    pub fn indices_mut(&mut self) -> &mut Indices { &mut self.indices }
    #[inline]
    pub fn sample(&self) -> &Sample { &self.sample }
    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample { &mut self.sample }

    #[must_use]
    #[inline]
    pub fn sample_vec(&mut self) -> Vec<usize> { self.sample.sort_to_vec() }

    #[must_use]
    #[inline]
    pub fn population_size(&self) -> usize
    where
        PROB: ProbabilityStore,
    {
        self.probabilities.len()
    }
    /// # Errors
    /// Returns an error if `ProbabilityStore` is empty
    #[inline]
    pub fn population_size_nz(&self) -> SamplingOptionsResult<NonZeroUsize>
    where
        PROB: ProbabilityStore,
    {
        NonZeroUsize::new(self.population_size()).ok_or(SamplingOptionsError::InvalidPopulationSize)
    }

    #[must_use]
    #[inline]
    pub fn draw<R>(
        &self,
        rng: &mut R,
        max: <PROB as ProbabilityStore>::PR,
    ) -> <PROB as ProbabilityStore>::PR
    where
        PROB: ProbabilityStore,
        R: RandomNumberGenerator,
    {
        self.probabilities.draw(rng, max)
    }
    #[inline]
    pub fn unit_decide(&mut self, idx: usize) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        match self.probabilities.unit_status(idx) {
            UnitDecisionStatus::In => {
                self.sample.add(idx);
                self.unit_remove(idx)?;
                Some(UnitDecisionStatus::In)
            }
            UnitDecisionStatus::Out => {
                self.unit_remove(idx)?;
                Some(UnitDecisionStatus::Out)
            }
            UnitDecisionStatus::Undecided => Some(UnitDecisionStatus::Undecided),
        }
    }
    #[inline]
    pub fn unit_set_and_decide(
        &mut self,
        idx: usize,
        prob: <PROB as ProbabilityStore>::PR,
    ) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        self.probabilities.set(idx, prob);
        self.unit_decide(idx)
    }
    #[inline]
    pub fn unit_set_max(&mut self, idx: usize) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        self.probabilities.set_max(idx);
        self.sample.add(idx);
        self.unit_remove(idx)?;
        Some(UnitDecisionStatus::In)
    }
    #[inline]
    pub fn unit_set_zero(&mut self, idx: usize) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        self.probabilities.set_zero(idx);
        self.unit_remove(idx)?;
        Some(UnitDecisionStatus::Out)
    }
    #[inline]
    pub fn unit_add_and_decide(
        &mut self,
        idx: usize,
        prob: <PROB as ProbabilityStore>::PR,
    ) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        self.probabilities.add(idx, prob);
        self.unit_decide(idx)
    }
    #[inline]
    pub fn unit_decide_last<R: RandomNumberGenerator>(
        &mut self,
        rng: &mut R,
    ) -> Option<UnitDecisionStatus>
    where
        Self: UnitRemoving,
        PROB: ProbabilityStore,
    {
        let Some(id) = self.indices.last() else {
            return Some(UnitDecisionStatus::Undecided);
        };
        let prob = self.probabilities.get(id);
        let max = self.probabilities.max();

        if self.draw(rng, max) < prob {
            self.unit_set_max(id)
        } else {
            self.unit_set_zero(id)
        }
    }
}

impl<PROB> SampleController<PROB, ()> {
    #[inline]
    pub fn new(probabilities: PROB) -> Self
    where
        PROB: ProbabilityStore,
    {
        let population_size = probabilities.len();
        let mut indices = Indices::new(population_size);
        let mut sample = Sample::new(population_size);

        // Reverse order guarantees units can be drawn from the back in order, see comment in
        // Indices::with_fill.
        for i in (0..population_size).rev() {
            match probabilities.unit_status(i) {
                UnitDecisionStatus::In => {
                    sample.add(i);
                }
                UnitDecisionStatus::Out => {}
                UnitDecisionStatus::Undecided => {
                    indices.insert(i);
                }
            };
        }

        Self {
            probabilities,
            indices,
            sample,
            tree: (),
        }
    }
    #[inline]
    pub fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        UnitRemoving::unit_remove(self, idx)
    }
}

impl<'bspread, PROB, P> SampleController<PROB, Tree<'bspread, P>>
where
    P: PointSet,
{
    #[inline]
    pub fn new_spreading(probabilities: PROB, spreading: &'bspread SpreadingOptions<P>) -> Self
    where
        PROB: ProbabilityStore,
    {
        let base_controller = SampleController::new(probabilities);
        let mut units = base_controller.indices.to_vec();
        let tree = Tree::new(spreading, &mut units);
        Self {
            probabilities: base_controller.probabilities,
            indices: base_controller.indices,
            sample: base_controller.sample,
            tree,
        }
    }
    #[inline]
    pub fn tree(&self) -> &Tree<'bspread, P> { &self.tree }
    #[inline]
    pub fn tree_mut(&mut self) -> &mut Tree<'bspread, P> { &mut self.tree }
    #[inline]
    pub fn reset_tree(&mut self, spreading: &'bspread SpreadingOptions<P>, units: &mut [usize]) {
        self.tree = Tree::new(spreading, units);
    }
    #[inline]
    pub fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        UnitRemoving::unit_remove(self, idx)
    }
}

pub trait UnitRemoving {
    fn unit_remove(&mut self, idx: usize) -> Option<usize>;
}
impl<PROB> UnitRemoving for SampleController<PROB, ()> {
    #[inline]
    fn unit_remove(&mut self, idx: usize) -> Option<usize> { self.indices.remove(idx) }
}
impl<PROB, P> UnitRemoving for SampleController<PROB, Tree<'_, P>>
where
    P: PointSet,
{
    #[inline]
    fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        self.tree.remove_unit(idx)?;
        self.indices.remove(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_options::SamplingOptions;
    use crate::test_utils::*;

    #[test]
    fn basic_controller_from_options() {
        let opts = SamplingOptions::with_spec(Data10::prob_u());
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
