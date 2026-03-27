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

use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::utils::usize_to_f64;

use super::energy_distance::EnergyDistance;
use crate::dbd::utils::gcd;

#[derive(Clone, Debug)]
/// Tactical configuration design parameters
pub struct TacticalConfigurationParameters {
    /// Population size of config
    population_size: usize,
    /// Sample size of config
    sample_size: usize,
    /// Number of samples (width) of config
    n_samples: usize,
    /// Number of times a unit exists in the config
    n_repeats: usize,
    // common_divisor: usize,
}
impl TacticalConfigurationParameters {
    /// Returns the population size
    pub fn population_size(&self) -> usize { self.population_size }
    /// Returns the sample size
    pub fn sample_size(&self) -> usize { self.sample_size }
    /// Returns the number of samples (width)
    pub fn n_samples(&self) -> usize { self.n_samples }
    /// Returns the multiplicity of the units
    pub fn n_repeats(&self) -> usize { self.n_repeats }

    /// Constructs a new config with given parameters
    pub fn new(
        population_size: usize,
        sample_size: usize,
        n_samples: usize,
        n_repeats: usize,
    ) -> Self {
        assert!(sample_size < population_size);
        assert_eq!(population_size * n_repeats, sample_size * n_samples);
        Self {
            population_size,
            sample_size,
            n_samples,
            n_repeats,
        }
    }
    /// Constructs a new minimum tactical configuration
    pub fn new_minimal(population_size: usize, sample_size: usize) -> Self {
        let common_divisor = gcd(population_size, sample_size);
        let n_samples = population_size / common_divisor;
        let n_repeats = sample_size / common_divisor;

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
    fn total_energy(&self) -> f64 { self.total_nenergy() / usize_to_f64(self.tcp().sample_size()) }
    /// Returns the average energy of all samples
    fn average_energy(&self) -> f64 { self.total_energy() / usize_to_f64(self.tcp().n_samples()) }
    /// Returns an iterator over a specific sample
    fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_;
    /// Returns an iterator over a random sample
    fn draw<R: RandomNumberGenerator>(
        &self,
        rng: &mut R,
    ) -> impl Iterator<Item = usize> + Clone + '_ {
        let n_samples = self.tcp().n_samples();
        let sample_id = rng.rusize_to(n_samples);
        self.sample(sample_id)
    }
    /// Returns the energy of a specific sample multiplied by the sample size
    fn nenergy_of_sample(&self, ed: &EnergyDistance, sample_id: usize) -> f64 {
        ed.total(self.sample(sample_id))
    }
}
