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
use envisim_utils::random::Rand;
use envisim_utils::sample_controller::{
    SampleController,
    TreeStorage,
};

/// A strategy for a pivotal method controls the selection of competing pairs
pub trait PivotalStrategy<PR, TR>
where
    PR: ProbabilityStore,
{
    /// Selects a pair of units
    #[inline]
    fn select_pair<R>(
        &mut self,
        _rng: &mut R,
        controller: &SampleController<PR, TR>,
    ) -> Pair<PR::Id>
    where
        R: Rand<usize>,
    {
        controller.indices().into()
    }
}

/// Runs a pivotal strategy
/// # Panics
/// Panics if options is incorrectly set up
#[inline]
pub fn pivotal_runner<R, PR, TR, S>(
    rng: &mut R,
    mut controller: SampleController<PR, TR>,
    mut strategy: S,
) -> SampleController<PR, TR>
where
    R: Rand<PR::N> + Rand<usize>,
    PR: ProbabilityStore,
    S: PivotalStrategy<PR, TR>,
    TR: TreeStorage<PR::Id>,
{
    loop {
        let (id1, id2) = match strategy.select_pair(rng, &controller) {
            Pair::More(id1, id2) | Pair::Two(id1, id2) => (id1, id2),
            Pair::One(_) => {
                let _last = controller.unit_decide_last(rng);
                return controller;
            }
            Pair::Zero => {
                return controller;
            }
        };

        let p1 = *controller.probabilities().get(id1).expect("id1 to exist");
        let p2 = *controller.probabilities().get(id2).expect("id2 to exist");
        let ctx = controller.probabilities().ctx();

        let (psum, prest) = {
            let mut ps = p1;
            let pr = ps.add(p2, ctx);
            (ps, pr)
        };

        if prest.is_zero(ctx) {
            // psum <= 1.0
            if controller.probabilities().draw_partial(rng, psum.get()) < p1 {
                controller.unit_set_and_decide(id1, psum);
                controller.unit_set_zero(id2);
            } else {
                controller.unit_set_zero(id1);
                controller.unit_set_and_decide(id2, psum);
            }
            // 1.0 < psum
        } else if controller
            .probabilities()
            .draw_partial(rng, *ctx.max() - prest.get())
            .get()
            < *ctx.max() - p2.get()
        {
            controller.unit_set_full(id1);
            controller.unit_set_and_decide(id2, prest);
        } else {
            controller.unit_set_and_decide(id1, prest);
            controller.unit_set_full(id2);
        }
    }
}
