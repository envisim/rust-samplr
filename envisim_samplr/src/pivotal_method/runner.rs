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
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::{
    SampleController,
    UnitRemoving,
};

pub trait PivotalStrategy<PST, TREE> {
    fn select_pair<R>(&mut self, controller: &mut SampleController<PST, TREE>, rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator;
}

#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "super is ok, needed for impl"
)]
#[must_use]
pub struct PivotalRunner<S, PST, TREE> {
    /// Sample controller
    pub(super) controller: SampleController<PST, TREE>,
    /// Sample strategy
    pub(super) strategy: S,
}
impl<S, PST, TREE> PivotalRunner<S, PST, TREE>
where
    S: PivotalStrategy<PST, TREE>,
    SampleController<PST, TREE>: UnitRemoving,
    PST: ProbabilityStore,
{
    /// Runs the simulation and returns a sorted sample
    #[must_use]
    #[inline]
    pub fn sample<R>(&mut self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        self.run(rng);
        self.controller.sample_vec()
    }
    /// Runs the sampling algorithm
    #[inline]
    pub fn run<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        while self.update_probabilities(rng) {}
        let _last = self.controller.unit_decide_last(rng);
    }
    /// Updates the probabilities according to the pivotal mehtod
    #[must_use]
    #[inline]
    fn update_probabilities<R>(&mut self, rng: &mut R) -> bool
    where
        R: RandomNumberGenerator,
    {
        let (id1, id2, cont) = match self.strategy.select_pair(&mut self.controller, rng) {
            Pair::More(id1, id2) => (id1, id2, true),
            Pair::Two(id1, id2) => (id1, id2, false),
            Pair::Zero | Pair::One(_) => {
                return false;
            }
        };

        let max = self.controller.probabilities().max();

        let p1 = self.controller.probabilities().data()[id1];
        let p2 = self.controller.probabilities().data()[id2];
        let psum = p1 + p2;

        if psum == max {
            if self.controller.draw(rng, max) < p1 {
                self.controller.unit_set_max(id1);
                self.controller.unit_set_zero(id2);
            } else {
                self.controller.unit_set_zero(id1);
                self.controller.unit_set_max(id2);
            }

            return cont;
        }

        if max < psum {
            if self.controller.draw(rng, max + max - psum) < max - p2 {
                self.controller.unit_set_max(id1);
                self.controller.unit_set_and_decide(id2, psum - max);
            } else {
                self.controller.unit_set_and_decide(id1, psum - max);
                self.controller.unit_set_max(id2);
            }
            return cont;
        }

        // psum < one
        if self.controller.probabilities().draw(rng, psum) < p1 {
            self.controller.unit_set_and_decide(id1, psum);
            self.controller.unit_set_zero(id2);
        } else {
            self.controller.unit_set_zero(id1);
            self.controller.unit_set_and_decide(id2, psum);
        }

        cont
    }
}
