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

use num_traits::{
    ConstZero,
    Signed,
};

use crate::indices::Indices;
use crate::kd_tree::{
    Tree,
    TreeResult,
};
use crate::probabilities::{
    ProbabilitySet,
    ProbabilityValue,
};
use crate::random::Rand;
use crate::sample::Sample;
use crate::sampling_options::SpreadingOptions;
use crate::utils::{
    ContiguousDataView,
    ContiguousPointSet,
    DataView,
    DataViewMut,
    PointSet,
    SliceView,
};

/// Provides method for removing unit(s) in `SampleController`.
pub trait TreeStorage<ID> {
    /// Removes unit `idx` from the storage
    #[inline]
    fn remove_unit(&mut self, _idx: ID) {}
}
impl<ID> TreeStorage<ID> for () {}
impl<P> TreeStorage<P::Id> for Tree<'_, P>
where
    P: PointSet,
{
    #[inline]
    fn remove_unit(&mut self, idx: P::Id) {
        let _exist: TreeResult<bool> = Tree::remove_unit(self, idx);
    }
}

/// A controller enabing algorithms to keep track of conditional probabilities, units, the sample
/// and some auxiliary information.
#[must_use]
#[derive(Debug, Clone)]
pub struct SampleController<PROB, TREE = ()>
where
    PROB: DataView<Value: ProbabilityValue>,
{
    /// The probability store
    probabilities: ProbabilitySet<PROB>,
    /// The (remaining) sample indices
    indices: Indices,
    /// The units included in the sample
    sample: Sample,
    /// The kd-tree containing the (remaining) units
    tree: TREE,
}

impl<PROB, TREE> SampleController<PROB, TREE>
where
    PROB: ContiguousDataView<Value: ProbabilityValue> + DataViewMut,
    TREE: TreeStorage<usize>,
{
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
    #[expect(clippy::missing_panics_doc, reason = "probs must not be empty")]
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.probabilities.len()).expect("probabilities to be non-empty")
    }
    /// Removes a unit if its probability is not partial.
    /// # Panics
    /// Panics if `idx` does not exist in `probabilities`.
    #[inline]
    pub fn unit_decide(&mut self, idx: PROB::Id) -> PROB::Value {
        let p = *self.probabilities.get(idx).expect("idx exist");
        if p.is_full(self.probabilities.ctx()) {
            self.sample.add(idx);
            self.unit_remove(idx);
        } else if p.is_zero(self.probabilities.ctx()) {
            self.unit_remove(idx);
        }
        p
    }
    /// Sets a unit to a new probability `prob` and removes it if `prob` is not partial.
    #[inline]
    pub fn unit_set_and_decide(&mut self, idx: PROB::Id, prob: PROB::Value) {
        self.probabilities.set(idx, prob);
        let _p = self.unit_decide(idx);
    }
    /// Sets a unit to a full probability representation and removes it
    #[inline]
    pub fn unit_set_full(&mut self, idx: PROB::Id) {
        self.probabilities.set_full(idx);
        self.sample.add(idx);
        self.unit_remove(idx);
    }
    /// Sets a unit to a zero probability representation and removes it
    #[inline]
    pub fn unit_set_zero(&mut self, idx: PROB::Id) {
        self.probabilities.set_zero(idx);
        self.unit_remove(idx);
    }
    /// Adds `prob` to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be added
    /// # Panics
    /// Panics if `idx` does not exist in `probabilities`.
    #[inline]
    pub fn unit_add_and_decide(&mut self, idx: PROB::Id, prob: PROB::Value) -> PROB::Value {
        let p = self.probabilities.add(idx, prob).expect("idx exist");
        let _decision_outcome = self.unit_decide(idx);
        p
    }
    /// Subtracts `prob` from the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be subtracted
    /// # Panics
    /// Panics if `idx` does not exist in `probabilities`.
    #[inline]
    pub fn unit_subtract_and_decide(&mut self, idx: PROB::Id, prob: PROB::Value) -> PROB::Value {
        let p = self.probabilities.subtract(idx, prob).expect("idx exist");
        let _decision_outcome = self.unit_decide(idx);
        p
    }
    /// Adds a delta to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount that could not be added
    /// # Panics
    /// Panics if delta (or -delta) cannot be constructed as a probability representation.
    #[inline]
    pub fn unit_add_delta_and_decide(
        &mut self,
        idx: PROB::Id,
        delta: <PROB::Value as ProbabilityValue>::N,
    ) -> PROB::Value
    where
        <PROB::Value as ProbabilityValue>::N: Signed,
    {
        if delta < <PROB::Value as ProbabilityValue>::N::ZERO {
            let p = ProbabilityValue::new(-delta, self.probabilities.ctx()).expect("delta is prob");
            self.unit_subtract_and_decide(idx, p)
        } else {
            let p = ProbabilityValue::new(delta, self.probabilities.ctx()).expect("delta is prob");
            self.unit_add_and_decide(idx, p)
        }
    }
    /// Decides the outcome of the last unit, or returns `None` if no last unit exists.
    #[expect(clippy::missing_panics_doc, reason = "panic is bug")]
    #[inline]
    pub fn unit_decide_last<R>(&mut self, rng: &mut R) -> Option<PROB::Value>
    where
        R: Rand<<PROB::Value as ProbabilityValue>::N>,
    {
        if self.indices.len() != 1 {
            return None;
        }
        let id = self.indices.last()?;
        let prob = *self.probabilities.get(id).expect("id exist");
        if self.probabilities.draw(rng) < prob {
            self.unit_set_full(id);
        } else {
            self.unit_set_zero(id);
        }
        Some(*self.probabilities.get(id).expect("id exist"))
    }
    /// Removes a unit from the controller
    #[inline]
    pub fn unit_remove(&mut self, idx: usize) -> bool {
        // Removing a non-existing unit seems like a bug, but should be caught by indices remove
        self.tree.remove_unit(idx);
        self.indices.remove(idx)
    }
}

impl<PROB> SampleController<PROB, ()>
where
    PROB: SliceView<Value: ProbabilityValue>,
{
    /// Constructs a new `SampleController` without a tree.
    /// # Panics
    /// Panics if probabilities is empty.
    #[inline]
    pub fn new(probabilities: ProbabilitySet<PROB>) -> Self {
        let population_size = probabilities.len();
        assert!(population_size > 0, "probabilities cannot be empty");
        let mut indices = Indices::new(population_size);
        let mut sample = Sample::new(population_size);

        // Reverse order guarantees units can be drawn from the back in order, see comment in
        // Indices::with_fill.
        let ctx = probabilities.ctx();
        for (i, p) in probabilities.slice().iter().enumerate().rev() {
            if p.is_full(ctx) {
                sample.add(i);
            } else if !p.is_zero(ctx) {
                assert!(
                    indices.insert(i),
                    "unit {i} should not already be in indices {indices:?}"
                );
            }
        }

        Self {
            probabilities,
            indices,
            sample,
            tree: (),
        }
    }
}

impl<'bspread, PROB, P> SampleController<PROB, Tree<'bspread, P>>
where
    PROB: SliceView<Value: ProbabilityValue>,
    P: ContiguousPointSet,
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
