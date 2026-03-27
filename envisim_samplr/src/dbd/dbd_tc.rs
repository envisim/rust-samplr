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

pub use config::*;
use envisim_utils::matrix::Matrix;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::utils::usize_to_f64;

use super::DistributionalDesignOptions;
use super::annealing::{
    AnnealingDistributionalDesign,
    AnnealingTemperature,
};
use super::energy_distance::EnergyDistance;
use crate::PivotalSampling;
use crate::utils::shuffled_indices;

mod config {
    use crate::dbd::energy_distance::EnergyDistance;
    pub use crate::dbd::tc_parameters::{
        DbdConfiguration,
        TacticalConfigurationParameters,
    };

    #[derive(Clone, Debug)]
    pub struct TacticalConfiguration {
        /// Internal storage of sequence (sample_size * n_samples matrix)
        // An n * M matrix
        buckets: Vec<usize>,
        /// Total energy multiplied by sample size
        total_nenergy: f64,
        /// Tactical configuration parameters
        tcp: TacticalConfigurationParameters,
    }
    impl TacticalConfiguration {
        /// Construct a new circular configuration from a set of indices and tactical configuration
        /// parameters
        pub fn new(
            sequence: Vec<usize>,
            tcp: TacticalConfigurationParameters,
            ed: &EnergyDistance,
        ) -> Self {
            assert_eq!(sequence.len(), tcp.n_samples() * tcp.sample_size(),);

            let mut cc = Self {
                buckets: sequence,
                total_nenergy: 0.0,
                tcp,
            };

            cc.reset_total_nenergy(ed);
            cc
        }
        /// Returns a reference to the internal storage
        pub fn buckets(&self) -> &[usize] { &self.buckets }
        /// Consumes `self` and returns the internal storage
        pub fn into_buckets(self) -> Vec<usize> { self.buckets }
        /// Returns a mutable reference to the internal storage
        pub fn buckets_mut(&mut self) -> &mut [usize] { &mut self.buckets }
        /// Returns the element at position `k` from a sample `sample_id`
        pub fn bucket_get(&self, sample_id: usize, k: usize) -> Option<usize> {
            self.buckets.get(self.index(sample_id, k)).cloned()
        }
        /// Returns the internal index of an element at position `k` in sample `sample_id`
        pub(crate) fn index(&self, sample_id: usize, k: usize) -> usize {
            assert!(sample_id < self.tcp.n_samples());
            assert!(k < self.tcp.sample_size());
            sample_id * self.tcp.sample_size() + k
        }
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
    impl DbdConfiguration for TacticalConfiguration {
        fn tcp(&self) -> &TacticalConfigurationParameters { &self.tcp }
        fn total_nenergy(&self) -> f64 { self.total_nenergy }
        fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_ {
            assert!(sample_id < self.tcp.n_samples());
            self.buckets
                .iter()
                .skip(sample_id * self.tcp.sample_size())
                .take(self.tcp.sample_size())
                .cloned()
        }
    }
}

#[derive(Clone, Debug)]
pub struct DbdTacticalConfiguration<'a> {
    temperature: AnnealingTemperature,
    ed: EnergyDistance<'a>,

    configuration: TacticalConfiguration,
    configuration_best: Option<TacticalConfiguration>,

    pair: ((usize, usize), (usize, usize)), // bucket index, k-index within bucket
    total_nenergy_delta: Option<f64>,
}

impl<'a> DbdTacticalConfiguration<'a> {
    pub fn optimal_configuration(&self) -> &TacticalConfiguration {
        self.configuration_best
            .as_ref()
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(&self.configuration)
    }
    pub fn into_optimal_configuration(self) -> TacticalConfiguration {
        self.configuration_best
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(self.configuration)
    }
    pub fn ed(&self) -> &EnergyDistance { &self.ed }
    pub fn tcp(&self) -> &TacticalConfigurationParameters { self.configuration.tcp() }

    // CONSTRUCTORS
    pub fn new<R: RandomNumberGenerator>(
        rng: &mut R,
        dbs_options: &DistributionalDesignOptions,
        matrix: &'a Matrix<'a>,
        sample_size: usize,
        eps: f64,
    ) -> Self {
        let population_size = matrix.nrow();
        let tcp = TacticalConfigurationParameters::new_minimal(population_size, sample_size);
        let annealing_temperature = dbs_options.as_annealing_temperature(eps);
        let ed = EnergyDistance::new(matrix, sample_size);

        let sequence = if dbs_options.spatial_initialization() {
            // Initial budget
            let mut b = vec![tcp.n_repeats(); tcp.population_size()];
            // Probability vector
            let mut p = vec![0.0f64; tcp.population_size()];
            // Offset to samples
            let mut offset = 0;
            let mut samples: Vec<usize> = vec![0; sample_size * tcp.n_samples()];

            for k in 0..tcp.n_samples() {
                // Set probabilities according to budget space
                for (id, &bb) in b.iter().enumerate() {
                    p[id] = if bb == 0 {
                        0.0
                    } else if bb == tcp.n_samples() {
                        1.0
                    } else {
                        usize_to_f64(bb) / usize_to_f64(tcp.n_samples() - k)
                    };
                }

                // Construct lpm opts
                let lpm_opts = SamplingOptions::new(&p)
                    .unwrap()
                    .set_spreading(matrix.clone_shallow())
                    .unwrap();
                let s = lpm_opts.lpm_2(rng).unwrap();
                assert_eq!(s.len(), tcp.sample_size());

                // For each unit in the sample, reduce the budget
                for (i, &id) in s.iter().enumerate() {
                    samples[offset + i] = id;
                    b[id] -= 1;
                }

                offset += tcp.sample_size();
            }

            samples
        } else {
            // Add random sequence in next version
            let sequence = shuffled_indices(rng, tcp.population_size());
            let mut samples: Vec<usize> = vec![0; sample_size * tcp.n_samples()];
            let mut i: usize = 0;
            for &id in sequence.iter() {
                for _ in 0..tcp.n_repeats() {
                    samples[(i % tcp.n_samples()) * sample_size + i / tcp.n_samples()] = id;
                    i += 1;
                }
            }

            samples
        };

        let pair = ((0, 0), (1, 0));
        let configuration = TacticalConfiguration::new(sequence, tcp, &ed);

        Self {
            temperature: annealing_temperature,
            ed,

            configuration,
            configuration_best: None,

            pair,
            total_nenergy_delta: None,
        }
    }
}

impl<'a> AnnealingDistributionalDesign for DbdTacticalConfiguration<'a> {
    fn temperature(&self) -> &AnnealingTemperature { &self.temperature }
    fn temperature_mut(&mut self) -> &mut AnnealingTemperature { &mut self.temperature }

    fn draw_units<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let sample_size = self.tcp().sample_size();

        loop {
            let a_unit = (
                rng.rusize_to(self.tcp().n_samples()),
                rng.rusize_to(sample_size),
            );
            let mut b_unit = (
                rng.rusize_to(self.tcp().n_samples() - 1),
                rng.rusize_to(sample_size),
            );

            if a_unit.0 == b_unit.0 {
                b_unit.0 = self.tcp().n_samples() - 1;
            }

            let a_id = self.configuration.bucket_get(a_unit.0, a_unit.1);
            let b_id = self.configuration.bucket_get(b_unit.0, b_unit.1);
            if a_id != b_id {
                self.pair = (a_unit, b_unit);
                self.total_nenergy_delta = None;
                return;
            }
        }
    }
    fn evaluate_switch(&mut self) -> Option<f64> {
        self.total_nenergy_delta = None;

        let ((gr1, k1), (gr2, k2)) = self.pair;
        let id1 = self.configuration.bucket_get(gr1, k1).unwrap();
        let id2 = self.configuration.bucket_get(gr2, k2).unwrap();

        let delta = self.ed.delta(self.configuration.sample(gr1), id2, id1)?
            + self.ed.delta(self.configuration.sample(gr2), id1, id2)?;

        self.total_nenergy_delta = Some(delta);
        self.total_nenergy_delta
    }
    fn switch(&mut self) {
        let Some(delta) = self.total_nenergy_delta else {
            return;
        };

        let k1 = self.configuration.index(self.pair.0.0, self.pair.0.1);
        let k2 = self.configuration.index(self.pair.1.0, self.pair.1.1);
        self.configuration.buckets_mut().swap(k1, k2);
        self.configuration.add_nenergy_delta(delta);
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
