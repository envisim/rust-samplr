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

//! Tactical configuration parameters

use std::num::NonZeroUsize;

use envisim_utils::random::Rand;
use envisim_utils::spatial::PointSet;
use num_integer::Integer;
use num_traits::ToPrimitive;

use super::energy_distance::EnergyDistance;

#[must_use]
#[derive(Clone, Debug)]
/// Tactical configuration design parameters
pub struct TacticalConfigurationParameters {
    /// Population size of config
    population_size: NonZeroUsize,
    /// Sample size of config
    sample_size: NonZeroUsize,
    /// Number of samples (width) of config
    n_samples: NonZeroUsize,
    /// Number of times a unit exists in the config
    n_repeats: NonZeroUsize,
    // common_divisor: NonZeroUsize,
}
impl TacticalConfigurationParameters {
    /// Returns the population size
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize { self.population_size }
    /// Returns the sample size
    #[must_use]
    #[inline]
    pub fn sample_size(&self) -> NonZeroUsize { self.sample_size }
    /// Returns the number of samples (width)
    #[must_use]
    #[inline]
    pub fn n_samples(&self) -> NonZeroUsize { self.n_samples }
    /// Returns the multiplicity of the units
    #[must_use]
    #[inline]
    pub fn n_repeats(&self) -> NonZeroUsize { self.n_repeats }

    /// Constructs a new config with given parameters
    ///
    /// # Panics
    /// Panics if `sample_size` is not lower than `population_size`, or if
    /// `population_size * n_repeats` does not equal `sample_size * n_samples`
    #[inline]
    pub fn new(
        population_size: NonZeroUsize,
        sample_size: NonZeroUsize,
        n_samples: NonZeroUsize,
        n_repeats: NonZeroUsize,
    ) -> Self {
        assert!(
            sample_size < population_size,
            "sample_size need to be lower than population_size"
        );
        assert_eq!(
            population_size.get() * n_repeats.get(),
            sample_size.get() * n_samples.get(),
            "invalid tc params"
        );
        Self {
            population_size,
            sample_size,
            n_samples,
            n_repeats,
        }
    }
    /// Constructs a new minimum tactical configuration
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug in gcd")]
    #[inline]
    pub fn new_minimal(population_size: NonZeroUsize, sample_size: NonZeroUsize) -> Self {
        let common_divisor = NonZeroUsize::new(population_size.get().gcd(&sample_size.get()))
            .expect("common divisor to be positive");
        let n_samples = NonZeroUsize::new(population_size.get() / common_divisor)
            .expect("pop size to be divisable by cd");
        let n_repeats = NonZeroUsize::new(sample_size.get() / common_divisor)
            .expect("sample size to be divisable by cd");

        Self {
            population_size,
            sample_size,
            n_samples,
            n_repeats,
            // common_divisor,
        }
    }
}

pub trait DbdConfiguration {
    /// Returns a reference to the tactical configuration parameters
    fn tcp(&self) -> &TacticalConfigurationParameters;
    /// Returns the total energy of all samples multiplied by the sample size
    #[must_use]
    fn total_nenergy(&self) -> f64;
    /// Returns the total energy of all samples
    #[must_use]
    #[inline]
    fn total_energy(&self) -> f64 {
        self.total_nenergy()
            / self
                .tcp()
                .sample_size()
                .get()
                .to_f64()
                .expect("sample_size to convert to f64")
    }
    /// Returns the average energy of all samples
    #[must_use]
    #[inline]
    fn average_energy(&self) -> f64 {
        self.total_energy()
            / self
                .tcp()
                .n_samples()
                .get()
                .to_f64()
                .expect("sample_size to convert to f64")
    }
    /// Returns an iterator over a specific sample
    #[must_use]
    fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_;
    /// Returns an iterator over a random sample
    #[must_use]
    #[inline]
    fn draw<R>(&self, rng: &mut R) -> impl Iterator<Item = usize> + Clone + '_
    where
        R: Rand<usize>,
    {
        let n_samples = self.tcp().n_samples().get();
        let sample_id = rng.rand_in(0..n_samples);
        self.sample(sample_id)
    }
    /// Returns the energy of a specific sample multiplied by the sample size
    #[must_use]
    #[inline]
    fn nenergy_of_sample<P>(&self, ed: &EnergyDistance<P>, sample_id: usize) -> f64
    where
        P: PointSet<N = f64>,
    {
        ed.total(self.sample(sample_id))
    }
}
