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

use std::num::NonZeroUsize;

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::spatial::PointSet;
use envisim_utils::utils::usize_to_f64;

use super::energy_distance::EnergyDistance;
use crate::dbd::utils::gcd;

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
    pub fn population_size(&self) -> NonZeroUsize { self.population_size }
    /// Returns the sample size
    pub fn sample_size(&self) -> NonZeroUsize { self.sample_size }
    /// Returns the number of samples (width)
    pub fn n_samples(&self) -> NonZeroUsize { self.n_samples }
    /// Returns the multiplicity of the units
    pub fn n_repeats(&self) -> NonZeroUsize { self.n_repeats }

    /// Constructs a new config with given parameters
    pub fn new(
        population_size: NonZeroUsize,
        sample_size: NonZeroUsize,
        n_samples: NonZeroUsize,
        n_repeats: NonZeroUsize,
    ) -> Self {
        assert!(sample_size < population_size);
        assert_eq!(
            population_size.get() * n_repeats.get(),
            sample_size.get() * n_samples.get()
        );
        Self {
            population_size,
            sample_size,
            n_samples,
            n_repeats,
        }
    }
    /// Constructs a new minimum tactical configuration
    pub fn new_minimal(population_size: NonZeroUsize, sample_size: NonZeroUsize) -> Self {
        let common_divisor = NonZeroUsize::new(gcd(population_size.get(), sample_size.get()))
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
    fn total_nenergy(&self) -> f64;
    /// Returns the total energy of all samples
    fn total_energy(&self) -> f64 {
        self.total_nenergy() / usize_to_f64(self.tcp().sample_size().get())
    }
    /// Returns the average energy of all samples
    fn average_energy(&self) -> f64 {
        self.total_energy() / usize_to_f64(self.tcp().n_samples().get())
    }
    /// Returns an iterator over a specific sample
    fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_;
    /// Returns an iterator over a random sample
    fn draw<R: RandomNumberGenerator>(
        &self,
        rng: &mut R,
    ) -> impl Iterator<Item = usize> + Clone + '_ {
        let n_samples = self.tcp().n_samples().get();
        let sample_id = rng.rusize_to(n_samples);
        self.sample(sample_id)
    }
    /// Returns the energy of a specific sample multiplied by the sample size
    fn nenergy_of_sample<P>(&self, ed: &EnergyDistance<P>, sample_id: usize) -> f64
    where
        P: PointSet<f64>,
    {
        ed.total(self.sample(sample_id))
    }
}
