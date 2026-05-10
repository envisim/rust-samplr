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
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
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
    pub fn new<PS, AUX, BAL>(
        options: &SamplingOptions<PS, AUX, BAL>,
    ) -> PivotalRunner<Self, PS::Native, ()>
    where
        PS: ProbabilitySpec,
    {
        let controller = options.to_controller();
        PivotalRunner {
            controller,
            strategy: Self(),
        }
    }
}
impl<PST> PivotalStrategy<PST, ()> for SequentialStrategy
where
    PST: ProbabilityStore,
{
    #[inline]
    fn select_pair<R>(&mut self, controller: &mut SampleController<PST, ()>, _rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator,
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
    pub fn new<PS, AUX, BAL>(
        options: &SamplingOptions<PS, AUX, BAL>,
    ) -> PivotalRunner<Self, PS::Native, ()>
    where
        PS: ProbabilitySpec,
    {
        let controller = options.to_controller();
        PivotalRunner {
            controller,
            strategy: Self(),
        }
    }
}
impl<PST> PivotalStrategy<PST, ()> for RandomStrategy
where
    PST: ProbabilityStore,
{
    #[inline]
    fn select_pair<R>(&mut self, controller: &mut SampleController<PST, ()>, rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator,
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
        let k = rng.rusize_to(len - 1);
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

pub trait PivotalSampling {
    fn spm<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
    fn rpm<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
}

impl<PS, AUX, BAL> PivotalSampling for SamplingOptions<PS, AUX, BAL>
where
    PS: ProbabilitySpec,
{
    /// Draw a sample using the sequential pivotal method.
    /// A variant of the pivotal method where unit competes in order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(p.into())?;
    /// let s = opts.spm(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tille, Y. (1998).
    /// Unequal probability sampling without replacement through a splitting method.
    /// Biometrika, 85(1), 89-101.
    /// <https://doi.org/10.1093/biomet/85.1.89>
    #[inline]
    fn spm<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        SequentialStrategy::new(self).sample(rng)
    }
    /// Draw a sample using the random pivotal method.
    /// A variant of the pivotal method where unit competes in a random order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = SmallRng::try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(p.into())?;
    /// let s = opts.rpm(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tille, Y. (1998).
    /// Unequal probability sampling without replacement through a splitting method.
    /// Biometrika, 85(1), 89-101.
    /// <https://doi.org/10.1093/biomet/85.1.89>
    #[inline]
    fn rpm<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        RandomStrategy::new(self).sample(rng)
    }
}
