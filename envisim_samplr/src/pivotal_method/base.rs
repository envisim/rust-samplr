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

//! Basic pivotal methods

use envisim_utils::indices::Pair;
use envisim_utils::random::{
    Rand,
    Rng,
};
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    ProbabilityOptions,
    SamplingOptions,
    SamplingOptionsRng,
};

use super::runner::{
    PivotalRunner,
    PivotalStrategy,
};

#[must_use]
pub struct SequentialStrategy();
impl SequentialStrategy {
    /// Constructs a new [`PivotalRunner`] using the sequential strategy
    #[inline]
    pub fn new<PO, AUX, BAL>(
        options: &SamplingOptions<PO, AUX, BAL>,
    ) -> PivotalRunner<Self, PO::Native, ()>
    where
        PO: ProbabilityOptions,
    {
        let controller = options.to_controller();
        PivotalRunner {
            controller,
            strategy: Self(),
        }
    }
}
impl<PROB> PivotalStrategy<PROB, ()> for SequentialStrategy {
    #[inline]
    fn select_pair<R>(&mut self, controller: &mut SampleController<PROB, ()>, _rng: &mut R) -> Pair
    where
        R: Rand<usize>,
    {
        // If Indices is initialized in reverse order, last units should be able to swap out safely,
        // so pairs are always in correct order
        controller.indices().into()
    }
}

pub struct RandomStrategy();
impl RandomStrategy {
    /// Constructs a new [`PivotalRunner`] using the random strategy
    #[inline]
    pub fn new<PO, AUX, BAL>(
        options: &SamplingOptions<PO, AUX, BAL>,
    ) -> PivotalRunner<Self, PO::Native, ()>
    where
        PO: ProbabilityOptions,
    {
        let controller = options.to_controller();
        PivotalRunner {
            controller,
            strategy: Self(),
        }
    }
}
impl<PROB> PivotalStrategy<PROB, ()> for RandomStrategy {
    #[inline]
    fn select_pair<R>(&mut self, controller: &mut SampleController<PROB, ()>, rng: &mut R) -> Pair
    where
        R: Rand<usize>,
    {
        let pair: Pair = controller.indices().into();
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

pub trait PivotalSampling<R>
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
    fn spm(&self, rng: &mut R) -> Vec<usize>;
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
    fn rpm(&self, rng: &mut R) -> Vec<usize>;
}

impl<R, PO, AUX, BAL> PivotalSampling<R> for SamplingOptions<PO, AUX, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions,
{
    #[inline]
    fn spm(&self, rng: &mut R) -> Vec<usize> { SequentialStrategy::new(self).sample(rng) }
    #[inline]
    fn rpm(&self, rng: &mut R) -> Vec<usize> { RandomStrategy::new(self).sample(rng) }
}
