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

//! Basic pivotal methods.

use envisim_utils::indices::Pair;
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::{
    Rand,
    Rng,
};
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsRng,
};
use envisim_utils::utils::ConstructableDataView;

use super::runner::{
    PivotalStrategy,
    pivotal_runner,
};

/// Sequential local pivotal method.
struct SequentialStrategy;
impl<PR> PivotalStrategy<PR, ()> for SequentialStrategy
where
    PR: ProbabilityStore,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        _rng: &mut R,
        controller: &SampleController<PR, ()>,
    ) -> Pair<<PR>::Id>
    where
        R: Rand<usize>,
    {
        // Empty means empty
        if controller.indices().is_empty() {
            return Pair::Zero;
        }

        let mut iter = controller
            .probabilities()
            .ids()
            .filter(|id| controller.indices().contains(*id));
        let pair = (iter.next(), iter.next());
        pair.into()
    }
}

/// Random order local pivotal method.
pub struct RandomStrategy;
impl<PR> PivotalStrategy<PR, ()> for RandomStrategy
where
    PR: ProbabilityStore,
{
    #[inline]
    fn select_pair<R>(&mut self, rng: &mut R, controller: &SampleController<PR, ()>) -> Pair<PR::Id>
    where
        R: Rand<usize>,
    {
        let pair: Pair<PR::Id> = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let len = controller.indices().len();
        let id1 = controller
            .indices()
            .draw(rng)
            .expect("indices to contain units");
        let k = rng.rand_to(len - 1);
        let mut id2 = controller.indices()[k];

        if id1 == id2 {
            id2 = controller
                .indices()
                .last()
                .expect("indices to contain units");
        }

        Pair::More(id1, id2)
    }
}

/// Provides pivotal sampling methods.
pub trait PivotalSampling<ID, R>
where
    R: Rng,
{
    /// Draw a sample using the sequential pivotal method.
    /// A variant of the pivotal method where unit competes in order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(p)?;
    /// let s = opts.spm(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn spm(&self, rng: &mut R) -> Vec<ID>;
    /// Draw a sample using the random pivotal method.
    /// A variant of the pivotal method where unit competes in a random order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(p)?;
    /// let s = opts.rpm(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    fn rpm(&self, rng: &mut R) -> Vec<ID>;
}

impl<R, PO, AUX, BAL> PivotalSampling<PO::Id, R> for SamplingOptions<PO, AUX, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec + ConstructableDataView,
{
    #[inline]
    fn spm(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new(self);
        pivotal_runner(rng, controller, SequentialStrategy).to_sorted_sample_vec()
    }
    #[inline]
    fn rpm(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new(self);
        pivotal_runner(rng, controller, RandomStrategy).to_sorted_sample_vec()
    }
}
