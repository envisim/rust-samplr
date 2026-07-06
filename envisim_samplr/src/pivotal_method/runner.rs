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

//! Runners and base traits for pivotal methods

use envisim_utils::indices::Pair;
use envisim_utils::random::Rand;
use envisim_utils::sample_controller::{
    SampleController,
    UnitRemoving,
};
use envisim_utils::utils::Number;

pub trait PivotalStrategy<PROB, TREE> {
    fn select_pair<R>(
        &mut self,
        controller: &mut SampleController<PROB, TREE>,
        rng: &mut R,
    ) -> Pair
    where
        R: Rand<usize>;
}

#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "super is ok, needed for impl"
)]
#[must_use]
pub struct PivotalRunner<S, PROB, TREE> {
    /// Sample controller
    pub(super) controller: SampleController<PROB, TREE>,
    /// Sample strategy
    pub(super) strategy: S,
}
impl<S, PROB, TREE> PivotalRunner<S, PROB, TREE>
where
    S: PivotalStrategy<PROB, TREE>,
    SampleController<PROB, TREE>: UnitRemoving,
    PROB: Number,
{
    /// Runs the simulation and returns a sorted sample
    #[must_use]
    #[inline]
    pub fn sample<R>(mut self, rng: &mut R) -> Vec<usize>
    where
        R: Rand<PROB>,
    {
        self.run(rng);
        self.controller.to_sorted_sample_vec()
    }
    /// Runs the sampling algorithm
    #[inline]
    pub fn run<R>(&mut self, rng: &mut R)
    where
        R: Rand<PROB>,
    {
        while self.update_probabilities(rng) {}
        let _last = self.controller.unit_decide_last(rng);
    }
    /// Updates the probabilities according to the pivotal mehtod
    #[must_use]
    #[inline]
    fn update_probabilities<R>(&mut self, rng: &mut R) -> bool
    where
        R: Rand<PROB>,
    {
        let (id1, id2, cont) = match self.strategy.select_pair(&mut self.controller, rng) {
            Pair::More(id1, id2) => (id1, id2, true),
            Pair::Two(id1, id2) => (id1, id2, false),
            Pair::Zero | Pair::One(_) => {
                return false;
            }
        };

        let max = self.controller.probabilities().max();
        let eps = self.controller.probabilities().eps();

        let p1 = self.controller.probabilities()[id1];
        let p2 = self.controller.probabilities()[id2];

        let (psum, prest) = {
            let mut ps = p1;
            let pr = ps.add(p2, max, eps);
            (ps, pr)
        };

        if prest.is_zero() {
            // psum <= 1.0
            if self
                .controller
                .probabilities()
                .draw_partial(rng, psum.get())
                < p1
            {
                self.controller.unit_set_and_decide(id1, psum);
                self.controller.unit_set_zero(id2);
            } else {
                self.controller.unit_set_zero(id1);
                self.controller.unit_set_and_decide(id2, psum);
            }
            // 1.0 < psum
        } else if self
            .controller
            .probabilities()
            .draw_partial(rng, max - prest.get())
            .get()
            < max - p2.get()
        {
            self.controller.unit_set_full(id1);
            self.controller.unit_set_and_decide(id2, prest);
        } else {
            self.controller.unit_set_and_decide(id1, prest);
            self.controller.unit_set_full(id2);
        }

        cont
    }
}
