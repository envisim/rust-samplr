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
    Probability,
    ProbabilitySet,
    ProbabilityStore,
};
use crate::random::Rand;
use crate::sample::Sample;
use crate::sampling_options::{
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsError,
    SamplingOptionsResult,
    SpreadingOptions,
};
use crate::utils::{
    ConstructableDataView,
    DataView,
    DataViewMut,
    Number,
    PointSet,
};

/// Provides method for removing unit(s) in `SampleController`.
pub trait TreeStorage<ID> {
    /// Removes unit `idx` from the storage
    #[inline]
    fn remove_unit(&mut self, _idx: ID) {}
}
impl<ID> TreeStorage<ID> for () {}
impl<DT> TreeStorage<DT::Id> for Tree<'_, DT>
where
    DT: PointSet,
{
    #[inline]
    fn remove_unit(&mut self, idx: DT::Id) {
        let _exist: TreeResult<bool> = Tree::remove_unit(self, idx);
    }
}

/// A controller enabing algorithms to keep track of conditional probabilities, units, the sample
/// and some auxiliary information.
#[must_use]
#[derive(Debug, Clone)]
pub struct SampleController<PR, TR = ()>
where
    PR: ProbabilityStore,
{
    /// The probability store
    probabilities: PR,
    /// The (remaining) sample indices
    indices: Indices<PR::Id>,
    /// The units included in the sample
    sample: Sample<PR::Id>,
    /// The kd-tree containing the (remaining) units
    tree: TR,
}

impl<PR, TR> SampleController<PR, TR>
where
    PR: ProbabilityStore,
{
    /// Returns a reference to the probability set
    #[inline]
    pub fn probabilities(&self) -> &PR { &self.probabilities }
    /// Returns a mutable reference to the probability set
    #[inline]
    pub fn probabilities_mut(&mut self) -> &mut PR { &mut self.probabilities }
    /// Returns a reference to the indices
    #[inline]
    pub fn indices(&self) -> &Indices<PR::Id> { &self.indices }
    /// Returns a mutable reference to the indices
    #[inline]
    pub fn indices_mut(&mut self) -> &mut Indices<PR::Id> { &mut self.indices }
    /// Returns a reference to the sample
    #[inline]
    pub fn sample(&self) -> &Sample<PR::Id> { &self.sample }
    /// Returns a mutable reference to the sample
    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample<PR::Id> { &mut self.sample }
    /// Returns a reference to the tree
    #[inline]
    pub fn tree(&self) -> &TR { &self.tree }
    /// Returns a mutable reference to the tree
    #[inline]
    pub fn tree_mut(&mut self) -> &mut TR { &mut self.tree }
    /// Moves self and returns the sorted vector of sample indices
    #[must_use]
    #[inline]
    pub fn to_sorted_sample_vec(self) -> Vec<PR::Id> { self.sample.to_sorted_vec() }
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
    pub fn unit_decide(&mut self, id: PR::Id) -> PR::Value
    where
        TR: TreeStorage<PR::Id>,
    {
        let p = *self.probabilities.get(id).expect("id exist");
        if p.is_full(self.probabilities.ctx()) {
            self.sample.add(id);
            self.unit_remove(id);
        } else if p.is_zero(self.probabilities.ctx()) {
            self.unit_remove(id);
        }
        p
    }
    /// Sets a unit to a new probability `prob` and removes it if `prob` is not partial.
    #[inline]
    pub fn unit_set_and_decide(&mut self, id: PR::Id, prob: PR::Value)
    where
        TR: TreeStorage<PR::Id>,
    {
        self.probabilities.set(id, prob);
        let _p = self.unit_decide(id);
    }
    /// Sets a unit to a full probability representation and removes it
    #[inline]
    pub fn unit_set_full(&mut self, id: PR::Id)
    where
        TR: TreeStorage<PR::Id>,
    {
        self.probabilities.set_full(id);
        self.sample.add(id);
        self.unit_remove(id);
    }
    /// Sets a unit to a zero probability representation and removes it
    #[inline]
    pub fn unit_set_zero(&mut self, id: PR::Id)
    where
        TR: TreeStorage<PR::Id>,
    {
        self.probabilities.set_zero(id);
        self.unit_remove(id);
    }
    /// Adds `prob` to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be added
    /// # Panics
    /// Panics if `idx` does not exist in `probabilities`.
    #[inline]
    pub fn unit_add_and_decide(&mut self, id: PR::Id, prob: PR::Value) -> PR::Value
    where
        TR: TreeStorage<PR::Id>,
    {
        let p = self.probabilities.add(id, prob).expect("idx exist");
        let _decision_outcome = self.unit_decide(id);
        p
    }
    /// Subtracts `prob` from the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount of `prob` that could not be subtracted
    /// # Panics
    /// Panics if `idx` does not exist in `probabilities`.
    #[inline]
    pub fn unit_subtract_and_decide(&mut self, id: PR::Id, prob: PR::Value) -> PR::Value
    where
        TR: TreeStorage<PR::Id>,
    {
        let p = self.probabilities.subtract(id, prob).expect("idx exist");
        let _decision_outcome = self.unit_decide(id);
        p
    }
    /// Adds a delta to the probability of a unit and removes it if the new sum is not partial.
    /// Returns the amount that could not be added
    /// # Panics
    /// Panics if delta (or -delta) cannot be constructed as a probability representation.
    #[inline]
    pub fn unit_add_delta_and_decide(&mut self, id: PR::Id, delta: PR::N) -> PR::Value
    where
        PR::N: Signed,
        TR: TreeStorage<PR::Id>,
    {
        if delta < <PR::N as ConstZero>::ZERO {
            let p = Probability::new(-delta, self.probabilities.ctx()).expect("delta is prob");
            self.unit_subtract_and_decide(id, p)
        } else {
            let p = Probability::new(delta, self.probabilities.ctx()).expect("delta is prob");
            self.unit_add_and_decide(id, p)
        }
    }
    /// Decides the outcome of the last unit, or returns `None` if no last unit exists.
    #[expect(clippy::missing_panics_doc, reason = "panic is bug")]
    #[inline]
    pub fn unit_decide_last<R>(&mut self, rng: &mut R) -> Option<PR::Value>
    where
        R: Rand<PR::N>,
        TR: TreeStorage<PR::Id>,
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
    pub fn unit_remove(&mut self, id: PR::Id) -> bool
    where
        TR: TreeStorage<PR::Id>,
    {
        // Removing a non-existing unit seems like a bug, but should be caught by indices remove
        self.tree.remove_unit(id);
        self.indices.remove(id)
    }
}

impl<T, N> SampleController<ProbabilitySet<T, N>, ()>
where
    T: DataViewMut<Value = Probability<N>>,
    N: Number,
{
    /// Constructs a new `SampleController` without a tree.
    /// # Errors
    /// If probabilities is empty.
    /// # Panics
    /// If probabilities contains non-unique ids.
    #[inline]
    pub fn from_probabilityset(probabilities: ProbabilitySet<T, N>) -> SamplingOptionsResult<Self> {
        if probabilities.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        let population_size = probabilities.len();
        let mut sample = Sample::new(population_size);
        let mut indices = Indices::try_from_iter(probabilities.ids()).expect("ids to be unique");

        let ctx = probabilities.ctx();
        for (id, p) in probabilities.entries() {
            if p.is_full(ctx) {
                sample.add(id);
            } else if !p.is_zero(ctx) {
                continue;
            }
            indices.remove(id);
        }

        Ok(Self {
            probabilities,
            indices,
            sample,
            tree: (),
        })
    }
    /// Constructs a new `SampleController` from options.
    #[expect(
        clippy::missing_panics_doc,
        reason = "popsize > 0 guaranteed by options"
    )]
    #[inline]
    pub fn new<PO, AUX, BAL>(opts: &SamplingOptions<PO, AUX, BAL>) -> Self
    where
        PO: ProbabilitiesSpec<Id = T::Id, Value = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let probs = opts.to_probabilityset();
        Self::from_probabilityset(probs).expect("population size > 0")
    }
    /// Constructs a new real-valued `SampleController` from options.
    #[expect(
        clippy::missing_panics_doc,
        reason = "popsize > 0 guaranteed by options"
    )]
    #[inline]
    pub fn new_real<PO, AUX, BAL>(opts: &SamplingOptions<PO, AUX, BAL>) -> Self
    where
        PO: ProbabilitiesSpec<Id = T::Id, Real = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let probs = opts.to_probabilityset_real();
        Self::from_probabilityset(probs).expect("population size > 0")
    }
}

impl<'bspread, T, N, DT> SampleController<ProbabilitySet<T, N>, Tree<'bspread, DT>>
where
    T: DataViewMut<Value = Probability<N>>,
    N: Number,
    DT: PointSet<Id = T::Id>,
{
    /// Constructs a new `SampleController` with a tree.
    /// # Panics
    /// Panics if ids are not unique.
    #[inline]
    pub fn new_spreading<PO, BAL>(
        opts: &'bspread SamplingOptions<PO, SpreadingOptions<DT>, BAL>,
    ) -> Self
    where
        PO: ProbabilitiesSpec<Id = T::Id, Value = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let base_controller = SampleController::new(opts);
        let mut units = base_controller.indices.to_vec();
        let tree = Tree::new(opts.spreading(), &mut units).expect("unique ids");
        Self {
            probabilities: base_controller.probabilities,
            indices: base_controller.indices,
            sample: base_controller.sample,
            tree,
        }
    }
    /// Constructs a new real-valued `SampleController` with a tree.
    /// # Panics
    /// Panics if ids are not unique.
    #[inline]
    pub fn new_real_spreading<PO, BAL>(
        opts: &'bspread SamplingOptions<PO, SpreadingOptions<DT>, BAL>,
    ) -> Self
    where
        PO: ProbabilitiesSpec<Id = T::Id, Real = N>
            + ConstructableDataView<ConstructableContainer<Probability<N>> = T>,
    {
        let base_controller = SampleController::new_real(opts);
        let mut units = base_controller.indices.to_vec();
        let tree = Tree::new(opts.spreading(), &mut units).expect("unique ids");
        Self {
            probabilities: base_controller.probabilities,
            indices: base_controller.indices,
            sample: base_controller.sample,
            tree,
        }
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
        let controller = SampleController::new(&opts);
        assert_eq!(controller.population_size().get(), 10);
    }

    #[test]
    fn basic_controller_exact() {
        let opts = SamplingOptions::new_equal(10, 3).unwrap();
        let controller = SampleController::new(&opts);
        assert_eq!(controller.population_size().get(), 10);
        assert_eq!(controller.probabilities().ctx().max(), &10);
    }
}
