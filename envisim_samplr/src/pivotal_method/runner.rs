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

use envisim_utils::indices::Pair;
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::SampleController;

pub trait PivotalStrategy<C> {
    fn select_pair<R>(&mut self, controller: &mut C, rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator;
}

pub struct PivotalRunner<C, S> {
    pub(super) controller: C,
    pub(super) strategy: S,
}
impl<C, ST, S> PivotalRunner<C, S>
where
    C: SampleController<Store = ST>,
    ST: ProbabilityStore,
    S: PivotalStrategy<C>,
{
    pub fn sample<R>(&mut self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        self.run(rng);
        self.controller.sample_mut().sort_to_vec()
    }
    pub fn run<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        while self.update_probabilities(rng) {}

        self.controller
            .unit_decide_last(rng)
            .expect("last unit to be decided");
    }
    fn update_probabilities<R>(&mut self, rng: &mut R) -> bool
    where
        R: RandomNumberGenerator,
    {
        let (id1, id2, cont) = match self.strategy.select_pair(&mut self.controller, rng) {
            Pair::More(id1, id2) => (id1, id2, true),
            Pair::Two(id1, id2) => (id1, id2, false),
            _ => {
                return false;
            }
        };

        let max = self.controller.probabilities().max();

        let p1 = self.controller.probabilities().data()[id1];
        let p2 = self.controller.probabilities().data()[id2];
        let psum = p1 + p2;

        if psum == max {
            if self.controller.draw(rng, max) < p1 {
                self.controller.unit_set_max(id1).expect("id1 to update");
                self.controller.unit_set_zero(id2).expect("id2 to update");
            } else {
                self.controller.unit_set_zero(id1).expect("id1 to update");
                self.controller.unit_set_max(id2).expect("id2 to update");
            }

            return cont;
        }

        if max < psum {
            if self.controller.draw(rng, max + max - psum) < max - p2 {
                self.controller.unit_set_max(id1).expect("id1 to update");
                self.controller
                    .unit_set_and_decide(id2, psum - max)
                    .expect("id2 to update");
            } else {
                self.controller
                    .unit_set_and_decide(id1, psum - max)
                    .expect("id1 to update");
                self.controller.unit_set_max(id2).expect("id2 to update");
            }
            return cont;
        }

        // psum < one
        if self.controller.probabilities().draw(rng, psum) < p1 {
            self.controller
                .unit_set_and_decide(id1, psum)
                .expect("id1 to update");
            self.controller.unit_set_zero(id2).expect("id2 to update");
        } else {
            self.controller.unit_set_zero(id1).expect("id1 to update");
            self.controller
                .unit_set_and_decide(id2, psum)
                .expect("id2 to update");
        }

        cont
    }
}
