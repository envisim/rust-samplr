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
use envisim_utils::sample_controller::{
    BasicSampleController,
    SampleController,
};
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
};

use super::runner::{
    PivotalRunner,
    PivotalStrategy,
};

pub struct SequentialStrategy {
    pair: (usize, usize),
}
impl SequentialStrategy {
    pub fn new<PS, SOP, M>(
        options: &SamplingOptions<'_, PS, SOP, M>,
    ) -> PivotalRunner<BasicSampleController<PS::Native>, Self>
    where
        PS: ProbabilitySpec,
    {
        let controller = options.to_controller();
        PivotalRunner {
            controller,
            strategy: Self { pair: (0, 1) },
        }
    }
}
impl<C, ST> PivotalStrategy<C> for SequentialStrategy
where
    C: SampleController<Store = ST>,
    ST: ProbabilityStore,
{
    fn select_pair<R>(&mut self, controller: &mut C, _rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_full() {
            return pair;
        }

        let pop_size = controller.population_size();

        // Check if the pair.0 was the unit to disappear
        if !controller.indices().contains(self.pair.0) {
            // Check if pair.1 also disappeared
            self.pair.0 = self.pair.1;

            if !controller.indices().contains(self.pair.0) {
                self.pair.0 = controller
                    .indices()
                    .seq_after(self.pair.0, pop_size)
                    .expect("two units to remain");
            }
        }

        // Now pair.0 is the first remaining unit...set pair.1 to the next remaining unit
        self.pair.1 = controller
            .indices()
            .seq_after(self.pair.0, pop_size)
            .expect("two units to remain");
        Pair::new(self.pair)
    }
}

pub struct RandomStrategy();
impl RandomStrategy {
    pub fn new<PS, SOP, M>(
        options: &SamplingOptions<'_, PS, SOP, M>,
    ) -> PivotalRunner<BasicSampleController<PS::Native>, Self>
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
impl<C, ST> PivotalStrategy<C> for RandomStrategy
where
    C: SampleController<Store = ST>,
    ST: ProbabilityStore,
{
    fn select_pair<R>(&mut self, controller: &mut C, rng: &mut R) -> Pair
    where
        R: RandomNumberGenerator,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let len = controller.indices().len();
        let id1 = controller.indices().draw(rng).unwrap();
        let k = rng.rusize_to(len - 1);
        let mut id2 = controller.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = controller.indices().last().unwrap();
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

impl<PS, SOP, M> PivotalSampling for SamplingOptions<'_, PS, SOP, M>
where
    PS: ProbabilitySpec,
{
    /// Draw a sample using the sequential pivotal method.
    /// A variant of the pivotal method where unit competes in order.
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(&p)?;
    /// let s = opts.spm(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tille, Y. (1998).
    /// Unequal probability sampling without replacement through a splitting method.
    /// Biometrika, 85(1), 89-101.
    /// <https://doi.org/10.1093/biomet/85.1.89>
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
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = vec![0.2f64, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(&p)?;
    /// let s = opts.rpm(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tille, Y. (1998).
    /// Unequal probability sampling without replacement through a splitting method.
    /// Biometrika, 85(1), 89-101.
    /// <https://doi.org/10.1093/biomet/85.1.89>
    fn rpm<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        RandomStrategy::new(self).sample(rng)
    }
}
