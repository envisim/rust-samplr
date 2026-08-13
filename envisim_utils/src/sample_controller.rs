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

//! Sample controller, utility function for sampling algorithms.

use std::num::NonZeroUsize;

use num_traits::Signed;

use crate::indices::Indices;
use crate::kd_tree::Tree;
use crate::probabilities::{
    Probability,
    ProbabilitySet,
};
use crate::random::Rand;
use crate::sample::Sample;
use crate::sampling_options::SpreadingOptions;
use crate::utils::{
    Number,
    PointSet,
};

/// A controller enabing algorithms to keep track of conditional probabilities, units, the sample
/// and some auxiliary information.
#[must_use]
#[derive(Debug, Clone)]
pub struct SampleController<PROB, TREE = ()> {
    /// The probability store
    probabilities: ProbabilitySet<PROB>,
    /// The (remaining) sample indices
    indices: Indices,
    /// The units included in the sample
    sample: Sample,
    /// The kd-tree containing the (remaining) units
    tree: TREE,
}

impl<PROB, TREE> SampleController<PROB, TREE> {
    /// Returns a reference to the probability set
    #[inline]
    pub fn probabilities(&self) -> &ProbabilitySet<PROB> { &self.probabilities }
    /// Returns a mutable reference to the probability set
    #[inline]
    pub fn probabilities_mut(&mut self) -> &mut ProbabilitySet<PROB> { &mut self.probabilities }
    /// Returns a reference to the indices
    #[inline]
    pub fn indices(&self) -> &Indices { &self.indices }
    /// Returns a mutable reference to the indices
    #[inline]
    pub fn indices_mut(&mut self) -> &mut Indices { &mut self.indices }
    /// Returns a reference to the sample
    #[inline]
    pub fn sample(&self) -> &Sample { &self.sample }
    /// Returns a mutable reference to the sample
    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample { &mut self.sample }
    /// Moves self and returns the sorted vector of sample indices
    #[must_use]
    #[inline]
    pub fn to_sorted_sample_vec(self) -> Vec<usize> { self.sample.to_sorted_vec() }
    /// Returns the populations size
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize { self.probabilities.len() }
    /// Removes a unit if its probability is not partial.
    #[inline]
    pub fn unit_decide(&mut self, idx: usize) -> Probability<PROB>
    where
        Self: UnitRemoving,
        PROB: Copy,
    {
        let p = self.probabilities[idx];
        match p {
            Probability::Partial(_) => {}
            Probability::Zero(_) => {
                self.unit_remove(idx);
            }
            Probability::Full(_) => {
                self.sample.add(idx);
                self.unit_remove(idx);
            }
        }
        p
    }
    /// Sets a unit to a new probability `prob` and removes it if `prob` is not partial.
    #[inline]
    pub fn unit_set_and_decide(&mut self, idx: usize, prob: Probability<PROB>)
    where
        Self: UnitRemoving,
        PROB: Number,
    {
        self.probabilities.set(idx, prob);
        let _p = self.unit_decide(idx);
    }
    /// Sets a unit to a full probability representation and removes it
    #[inline]
    pub fn unit_set_full(&mut self, idx: usize)
    where
        Self: UnitRemoving,
        PROB: Copy,
    {
        self.probabilities.set_full(idx);
        self.sample.add(idx);
        self.unit_remove(idx);
    }
    /// Sets a unit to a zero probability representation and removes it
    #[inline]
    pub fn unit_set_zero(&mut self, idx: usize)
    where
        Self: UnitRemoving,
        PROB: Number,
    {
        self.probabilities.set_zero(idx);
        self.unit_remove(idx);
    }
    /// Adds `prob` to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be added
    #[inline]
    pub fn unit_add_and_decide(&mut self, idx: usize, prob: Probability<PROB>) -> Probability<PROB>
    where
        Self: UnitRemoving,
        PROB: Number,
    {
        let p = self.probabilities.add(idx, prob);
        let _decision_outcome = self.unit_decide(idx);
        p
    }
    /// Subtracts `prob` from the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be subtracted
    #[inline]
    pub fn unit_subtract_and_decide(
        &mut self,
        idx: usize,
        prob: Probability<PROB>,
    ) -> Probability<PROB>
    where
        Self: UnitRemoving,
        PROB: Number,
    {
        let p = self.probabilities.subtract(idx, prob);
        let _decision_outcome = self.unit_decide(idx);
        p
    }
    /// Adds a delta to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount that could not be added
    #[inline]
    pub fn unit_add_delta_and_decide(&mut self, idx: usize, delta: PROB) -> Probability<PROB>
    where
        Self: UnitRemoving,
        PROB: Number + Signed,
    {
        if delta < PROB::ZERO {
            let p = Probability::Partial(-delta);
            self.unit_subtract_and_decide(idx, p)
        } else {
            let p = Probability::Partial(delta);
            self.unit_add_and_decide(idx, p)
        }
    }
    /// Decides the outcome of the last unit, or returns `None` if no last unit exists.
    #[inline]
    pub fn unit_decide_last<R>(&mut self, rng: &mut R) -> Option<Probability<PROB>>
    where
        Self: UnitRemoving,
        R: Rand<PROB>,
        PROB: Number,
    {
        if self.indices.len() != 1 {
            return None;
        }
        let id = self.indices.last()?;
        let prob = self.probabilities[id];
        if self.probabilities.draw(rng) < prob {
            self.unit_set_full(id);
        } else {
            self.unit_set_zero(id);
        }
        Some(self.probabilities[id])
    }
}

impl<PROB> SampleController<PROB, ()> {
    /// Constructs a new `SampleController` without a tree.
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
    #[inline]
    pub fn new(probabilities: ProbabilitySet<PROB>) -> Self {
        let population_size = probabilities.len().get();
        let mut indices = Indices::new(population_size);
        let mut sample = Sample::new(population_size);

        // Reverse order guarantees units can be drawn from the back in order, see comment in
        // Indices::with_fill.
        for i in (0..population_size).rev() {
            match probabilities[i] {
                Probability::Partial(_) => {
                    assert!(
                        indices.insert(i),
                        "unit {i} should not already be in indices {indices:?}"
                    );
                }
                Probability::Zero(_) => {}
                Probability::Full(_) => {
                    sample.add(i);
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
    /// Removes a unit from the controller
    #[inline]
    pub fn unit_remove(&mut self, idx: usize) -> bool { UnitRemoving::unit_remove(self, idx) }
}

impl<'bspread, PROB, P> SampleController<PROB, Tree<'bspread, P>>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new `SampleController` with a tree.
    #[expect(clippy::missing_panics_doc, reason = "units are based on self indices")]
    #[inline]
    pub fn new_spreading(
        probabilities: ProbabilitySet<PROB>,
        spreading: &'bspread SpreadingOptions<P>,
    ) -> Self {
        let base_controller = SampleController::new(probabilities);
        let mut units = base_controller.indices.to_vec();
        let tree = Tree::new(spreading, &mut units).expect("units based on indices to exist");
        Self {
            probabilities: base_controller.probabilities,
            indices: base_controller.indices,
            sample: base_controller.sample,
            tree,
        }
    }
    /// Returns a reference to the tree
    #[inline]
    pub fn tree(&self) -> &Tree<'bspread, P> { &self.tree }
    /// Returns a mutable reference to the tree
    #[inline]
    pub fn tree_mut(&mut self) -> &mut Tree<'bspread, P> { &mut self.tree }
    /// Removes a unit from the controller
    #[inline]
    pub fn unit_remove(&mut self, idx: usize) -> bool { UnitRemoving::unit_remove(self, idx) }
}

/// Trait for sample controllers that can remove a unit
pub trait UnitRemoving {
    /// Returns `true` if `idx` was present and removed, `false` if `idx` wasn't found.
    fn unit_remove(&mut self, idx: usize) -> bool;
}
impl<PROB> UnitRemoving for SampleController<PROB, ()> {
    #[inline]
    fn unit_remove(&mut self, idx: usize) -> bool { self.indices.remove(idx) }
}
impl<PROB, P> UnitRemoving for SampleController<PROB, Tree<'_, P>>
where
    P: PointSet<Id = usize>,
{
    #[inline]
    fn unit_remove(&mut self, idx: usize) -> bool {
        // Removing a non-existing unit seems like a bug, but should be caught by indices remove
        let _res = self.tree.remove_unit(idx);
        self.indices.remove(idx)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
    use crate::sampling_options::SamplingOptions;
    use crate::test_utils::*;

    #[test]
    fn basic_controller_from_options() {
        let opts = SamplingOptions::with_spec(Data10::prob_u());
        let controller = opts.to_controller();
        assert_eq!(controller.population_size().get(), 10);
    }

    #[test]
    fn basic_controller_exact() {
        let opts = SamplingOptions::new_equal(10, 3).unwrap();
        let controller = opts.to_controller();
        assert_eq!(controller.population_size().get(), 10);
        assert_eq!(controller.probabilities().max(), 10);
    }
}
