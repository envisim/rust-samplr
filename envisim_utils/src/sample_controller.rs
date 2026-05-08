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
use crate::number_traits::Number;
use crate::probabilities::ProbabilityStore;
use crate::random::RandomNumberGenerator;
use crate::sampling_options::{
    SamplingOptionsError,
    SamplingOptionsResult,
    SpreadingOptions,
};

/// Sample container
#[must_use]
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

/// Decision result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "a unit can only exists in three decision states"
)]
#[must_use]
pub enum DecideUnit {
    In,
    Out,
    Undecided,
}

pub trait SampleController {
    type Store: ProbabilityStore;
    fn controller(&self) -> &BasicSampleController<Self::Store>;
    fn controller_mut(&mut self) -> &mut BasicSampleController<Self::Store>;
    #[must_use]
    #[inline]
    fn probabilities(&self) -> &Self::Store { &self.controller().probabilities }
    #[must_use]
    #[inline]
    fn probabilities_mut(&mut self) -> &mut Self::Store { &mut self.controller_mut().probabilities }
    #[inline]
    fn indices(&self) -> &Indices { &self.controller().indices }
    #[inline]
    fn indices_mut(&mut self) -> &mut Indices { &mut self.controller_mut().indices }
    #[inline]
    fn sample(&self) -> &Sample { &self.controller().sample }
    #[inline]
    fn sample_mut(&mut self) -> &mut Sample { &mut self.controller_mut().sample }

    #[must_use]
    #[inline]
    fn population_size(&self) -> usize { self.probabilities().len() }
    /// # Errors
    /// Returns an error if `ProbabilityStore` is empty
    #[inline]
    fn population_size_nz(&self) -> SamplingOptionsResult<NonZeroUsize> {
        NonZeroUsize::new(self.population_size()).ok_or(SamplingOptionsError::InvalidPopulationSize)
    }

    #[must_use]
    #[inline]
    fn draw<R: RandomNumberGenerator>(
        &self,
        rng: &mut R,
        max: <Self::Store as ProbabilityStore>::PR,
    ) -> <Self::Store as ProbabilityStore>::PR {
        self.probabilities().draw(rng, max)
    }

    #[inline]
    fn unit_remove(&mut self, idx: usize) -> Option<usize> { self.indices_mut().remove(idx) }
    #[inline]
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
    #[inline]
    fn unit_set_and_decide(
        &mut self,
        idx: usize,
        prob: <Self::Store as ProbabilityStore>::PR,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().set(idx, prob);
        self.unit_decide(idx)
    }
    #[inline]
    fn unit_set_max(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_max(idx);
        self.sample_mut().add(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::In)
    }
    #[inline]
    fn unit_set_zero(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_zero(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::Out)
    }
    #[inline]
    fn unit_add_and_decide(
        &mut self,
        idx: usize,
        prob: <Self::Store as ProbabilityStore>::PR,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().add(idx, prob);
        self.unit_decide(idx)
    }
    #[inline]
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
#[must_use]
pub struct BasicSampleController<ST> {
    /// The probability store
    probabilities: ST,
    /// The (remaining) sample indices
    indices: Indices,
    /// The units included in the sample
    sample: Sample,
}
impl<ST> BasicSampleController<ST> {
    #[inline]
    pub fn new(probabilities: ST) -> Self
    where
        ST: ProbabilityStore,
    {
        let population_size = probabilities.len();
        let mut controller = Self {
            probabilities,
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };

        // Decide all units
        for i in 0..population_size {
            controller.unit_decide(i);
        }

        controller
    }
}

impl<ST> SampleController for BasicSampleController<ST>
where
    ST: ProbabilityStore,
{
    type Store = ST;
    #[inline]
    fn controller(&self) -> &BasicSampleController<ST> { self }
    #[inline]
    fn controller_mut(&mut self) -> &mut BasicSampleController<ST> { self }
    #[inline]
    fn probabilities(&self) -> &ST { &self.probabilities }
    #[inline]
    fn probabilities_mut(&mut self) -> &mut ST { &mut self.probabilities }
    #[inline]
    fn indices(&self) -> &Indices { &self.indices }
    #[inline]
    fn indices_mut(&mut self) -> &mut Indices { &mut self.indices }
    #[inline]
    fn sample(&self) -> &Sample { &self.sample }
    #[inline]
    fn sample_mut(&mut self) -> &mut Sample { &mut self.sample }
}

// SpreadingSampleController
#[must_use]
pub struct SpreadingSampleController<'bspread, ST, N, P> {
    /// The basic controller
    controller: BasicSampleController<ST>,
    /// The kd-tree containing the (remaining) units
    tree: Tree<'bspread, N, P>,
}

impl<'bspread, ST, N, P> SpreadingSampleController<'bspread, ST, N, P> {
    #[inline]
    pub fn new(
        controller: BasicSampleController<ST>,
        spreading: &'bspread SpreadingOptions<P>,
    ) -> Self
    where
        ST: ProbabilityStore,
        N: Number,
        P: PointSet<N>,
    {
        let mut units = controller.indices().to_vec();
        let tree = Tree::new(spreading, &mut units);
        Self { controller, tree }
    }
    #[inline]
    pub fn tree(&self) -> &Tree<'bspread, N, P> { &self.tree }
    #[inline]
    pub fn tree_mut(&mut self) -> &mut Tree<'bspread, N, P> { &mut self.tree }
    #[inline]
    pub fn reset_tree(&mut self, spreading: &'bspread SpreadingOptions<P>, units: &mut [usize])
    where
        N: Number,
        P: PointSet<N>,
    {
        self.tree = Tree::new(spreading, units);
    }
}

impl<ST, N, P> SampleController for SpreadingSampleController<'_, ST, N, P>
where
    ST: ProbabilityStore,
    N: Number,
    P: PointSet<N>,
{
    type Store = ST;
    #[inline]
    fn controller(&self) -> &BasicSampleController<ST> { &self.controller }
    #[inline]
    fn controller_mut(&mut self) -> &mut BasicSampleController<ST> { &mut self.controller }
    #[inline]
    fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        self.tree.remove_unit(idx)?;
        self.controller.unit_remove(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_options::SamplingOptions;
    use crate::test_utils::*;

    #[test]
    fn basic_controller_from_options() {
        let opts = SamplingOptions::with_spec(Data10::prob_u()).unwrap();
        let controller = opts.to_controller();
        assert_eq!(controller.population_size(), 10);
    }

    #[test]
    fn basic_controller_exact() {
        let opts = SamplingOptions::new_equal(nz(10), 3).unwrap();
        let controller = opts.to_controller();
        assert_eq!(controller.population_size(), 10);
        assert_eq!(controller.probabilities().max(), 10);
    }
}
