// Copyright (C) 2026 Wilmer Prentius, Anton Grafström.
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

pub use config::*;
use envisim_utils::matrix::Matrix;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::utils::usize_to_f64;

use super::DistributionalDesignOptions;
use super::annealing::{
    AnnealingDistributionalDesign,
    AnnealingTemperature,
};
use super::energy_distance::EnergyDistance;

mod config {
    use crate::dbd::energy_distance::EnergyDistance;
    pub use crate::dbd::tc_parameters::{
        DbdConfiguration,
        TacticalConfigurationParameters,
    };

    /// Stores a circular design configuration
    #[derive(Clone, Debug)]
    pub struct CircularConfiguration {
        /// Internal storage of sequence
        sequence: Vec<usize>,
        /// Total energy multiplied by sample size
        total_nenergy: f64,
        /// Tactical configuration parameters
        tcp: TacticalConfigurationParameters,
    }
    impl CircularConfiguration {
        /// Construct a new circular configuration from a sequence
        pub fn new(sequence: Vec<usize>, sample_size: usize, ed: &EnergyDistance) -> Self {
            let population_size = sequence.len();
            assert!(0 < population_size);
            assert!(0 < sample_size && sample_size < population_size);

            let mut cc = Self {
                sequence,
                total_nenergy: 0.0,
                tcp: TacticalConfigurationParameters::new(
                    population_size,
                    sample_size,
                    population_size,
                    sample_size,
                ),
            };

            cc.reset_total_nenergy(ed);
            cc
        }
        /// Returns a reference to the sequence store
        pub fn sequence(&self) -> &[usize] { &self.sequence }
        /// Consumes `self` and returns the internal storage
        pub fn into_sequence(self) -> Vec<usize> { self.sequence }
        /// Returns a mutable reference to the sequence store
        pub fn sequence_mut(&mut self) -> &mut [usize] { &mut self.sequence }
        /// Returns an element from the sequence store
        pub fn sequence_get(&self, k: usize) -> Option<usize> { self.sequence.get(k).cloned() }
        /// Add a delta to the nenergy
        pub(crate) fn add_nenergy_delta(&mut self, delta: f64) -> f64 {
            self.total_nenergy += delta;
            self.total_nenergy
        }
        /// Reset the total nenergy
        fn reset_total_nenergy(&mut self, ed: &EnergyDistance) -> f64 {
            self.total_nenergy = 0.0;
            for i in 0..self.tcp.n_samples() {
                self.total_nenergy += self.nenergy_of_sample(ed, i);
            }
            self.total_nenergy
        }
    }
    impl DbdConfiguration for CircularConfiguration {
        fn tcp(&self) -> &TacticalConfigurationParameters { &self.tcp }
        fn total_nenergy(&self) -> f64 { self.total_nenergy }
        fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_ {
            let sample_id = sample_id % self.tcp.n_samples();
            self.sequence[sample_id..]
                .iter()
                .chain(self.sequence[..sample_id].iter())
                .take(self.tcp.sample_size())
                .cloned()
        }
    }
}

pub struct DbdCircular<'a> {
    temperature: AnnealingTemperature,
    ed: EnergyDistance<'a>,

    configuration: CircularConfiguration,
    configuration_best: Option<CircularConfiguration>,

    pair: (usize, usize), // k-index
    total_nenergy_delta: f64,
}
impl<'a> DbdCircular<'a> {
    pub fn optimal_configuration(&self) -> &CircularConfiguration {
        self.configuration_best
            .as_ref()
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(&self.configuration)
    }
    pub fn into_optimal_configuration(self) -> CircularConfiguration {
        self.configuration_best
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(self.configuration)
    }
    pub fn ed(&self) -> &EnergyDistance { &self.ed }
    pub fn tcp(&self) -> &TacticalConfigurationParameters { self.configuration.tcp() }

    // CONSTRUCTORS
    pub fn new(
        dbs_options: &DistributionalDesignOptions,
        matrix: &'a Matrix<'a>,
        sample_size: usize,
        eps: f64,
    ) -> Self {
        let population_size = matrix.nrow();
        let sequence: Vec<usize> = (0..population_size).collect();
        let annealing_temperature = dbs_options.as_annealing_temperature(eps);
        let ed = EnergyDistance::new(matrix, sample_size);

        let pair = (sequence[0], sequence[1]);
        let configuration = CircularConfiguration::new(sequence, sample_size, &ed);

        Self {
            temperature: annealing_temperature,
            ed,

            configuration,
            configuration_best: None,

            pair,
            total_nenergy_delta: 0.0,
        }
    }
}

impl<'a> AnnealingDistributionalDesign for DbdCircular<'a> {
    fn temperature(&self) -> &AnnealingTemperature { &self.temperature }
    fn temperature_mut(&mut self) -> &mut AnnealingTemperature { &mut self.temperature }

    fn draw_units<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let n = self.tcp().population_size();
        let a = rng.rusize_to(n);
        let b = rng.rusize_to(n - 1);
        self.pair = if a == b { (a, n - 1) } else { (a, b) };
    }
    fn evaluate_switch(&mut self) -> Option<f64> {
        self.total_nenergy_delta = 0.0;

        // Start off by clearing deltas
        let (k1, k2) = self.pair;
        if k1 == k2 {
            return None;
        }

        let population_size = self.tcp().population_size();
        let sample_size = self.tcp().sample_size();
        let d = population_size - sample_size + 1;

        let id1 = self.configuration.sequence_get(k1).unwrap();
        let id2 = self.configuration.sequence_get(k2).unwrap();

        // Units definetly in the overlap should not be counted
        // Other units should be counted relative to their distance to the moving unit
        // Look in the window -sample_size +sample_size around k1
        for k in (k1 + d)..(k1 + d + sample_size * 2 - 1) {
            let k = k % population_size;
            if k == k1 || k == k2 {
                continue;
            }

            let abs_dist = k1.abs_diff(k);
            let dist = abs_dist.min(population_size - abs_dist);
            let m1: i64 = if dist < sample_size {
                i64::try_from(sample_size - dist).unwrap()
                // (sample_size - dist) as i64
            } else {
                0
            };

            let abs_dist = k2.abs_diff(k);
            let dist = abs_dist.min(population_size - abs_dist);
            let m2: i64 = if dist < sample_size {
                i64::try_from(sample_size - dist).unwrap()
                // (sample_size - dist) as i64
            } else {
                0
            };

            if m1 == m2 {
                continue;
            }

            let m_f64 = ((2 * (m1 - m2)) as f64) / usize_to_f64(sample_size);

            let id = self.configuration.sequence_get(k).unwrap();
            let delta = self.ed.relative_distance(id, id1, id2);
            self.total_nenergy_delta += delta * m_f64;
        }

        for k in (k2 + d)..(k2 + d + sample_size * 2 - 1) {
            let k = k % population_size;
            if k == k1 || k == k2 {
                continue;
            }

            // Skip every unit close to k1, as these has already been handled
            let abs_dist = k1.abs_diff(k);
            let dist = abs_dist.min(population_size - abs_dist);
            if dist < sample_size {
                continue;
            }

            let abs_dist = k2.abs_diff(k);
            let dist = abs_dist.min(population_size - abs_dist);
            let m_f64: f64 = if dist < sample_size {
                ((2 * (sample_size - dist)) as f64) / usize_to_f64(sample_size)
            } else {
                continue;
            };

            let id = self.configuration.sequence_get(k).unwrap();
            let delta = self.ed.relative_distance(id, id2, id1);
            self.total_nenergy_delta += delta * m_f64;
        }

        Some(self.total_nenergy_delta)
    }
    fn switch(&mut self) {
        let (k1, k2) = self.pair;
        self.configuration.sequence_mut().swap(k1, k2);
        self.configuration
            .add_nenergy_delta(self.total_nenergy_delta);
    }
    fn set_optimal_configuration(&mut self) {
        // Only replace if conf_best is None, or if we're better than conf_best
        self.configuration_best = match &self.configuration_best {
            // If there is a best, check if the current configuration is better
            Some(best) if self.configuration.total_nenergy() < best.total_nenergy() => {
                Some(self.configuration.clone())
            }
            // If there is no best yet
            None => Some(self.configuration.clone()),
            // Otherwise the current best is still better
            _ => None,
        };

        if self.configuration_best.is_some() {
            self.temperature.increment_divergence();
        }
    }
}
