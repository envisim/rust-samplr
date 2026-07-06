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

//! Circular distributionally balanced designs

use std::num::NonZeroUsize;

pub use config::*;
use envisim_utils::random::Rand;
use envisim_utils::utils::{
    Epsilon,
    PointSet,
};
use num_traits::ToPrimitive;

use super::DistributionalDesignOptions;
use super::annealing::{
    AnnealingDistributionalDesign,
    AnnealingTemperature,
};
use super::energy_distance::EnergyDistance;

mod config {
    //! Circular DBD config

    use std::num::NonZeroUsize;

    use envisim_utils::utils::PointSet;

    use crate::dbd::energy_distance::EnergyDistance;
    pub use crate::dbd::tc_parameters::{
        DbdConfiguration,
        TacticalConfigurationParameters,
    };

    /// Stores a circular design configuration
    #[must_use]
    #[derive(Clone, Debug)]
    pub struct CircularConfiguration {
        /// Internal storage of sequence
        sequence: Box<[usize]>,
        /// Total energy multiplied by sample size
        total_nenergy: f64,
        /// Tactical configuration parameters
        tcp: TacticalConfigurationParameters,
    }
    impl CircularConfiguration {
        /// Construct a new circular configuration from a sequence
        ///
        /// # Panics
        /// Panics if the sequence is empty.
        #[inline]
        pub fn new<P>(
            sequence: Box<[usize]>,
            sample_size: NonZeroUsize,
            ed: &EnergyDistance<P>,
        ) -> Self
        where
            P: PointSet<Id = usize, Value = f64>,
        {
            let population_size =
                NonZeroUsize::new(sequence.len()).expect("sequence to be non-empty");
            let sample_size = sample_size.min(population_size);

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
        #[must_use]
        #[inline]
        pub fn sequence(&self) -> &[usize] { &self.sequence }
        /// Consumes `self` and returns the internal storage
        #[must_use]
        #[inline]
        pub fn into_sequence(self) -> Box<[usize]> { self.sequence }
        /// Returns a mutable reference to the sequence store
        #[must_use]
        #[inline]
        pub fn sequence_mut(&mut self) -> &mut [usize] { &mut self.sequence }
        /// Returns an element from the sequence store
        #[must_use]
        #[inline]
        pub fn sequence_get(&self, k: usize) -> Option<usize> { self.sequence.get(k).copied() }
        /// Add a delta to the nenergy
        pub(crate) fn add_nenergy_delta(&mut self, delta: f64) -> f64 {
            self.total_nenergy += delta;
            self.total_nenergy
        }
        /// Reset the total nenergy
        #[inline]
        fn reset_total_nenergy<P>(&mut self, ed: &EnergyDistance<P>) -> f64
        where
            P: PointSet<Id = usize, Value = f64>,
        {
            self.total_nenergy = 0.0;
            for i in 0..self.tcp.n_samples().get() {
                self.total_nenergy += self.nenergy_of_sample(ed, i);
            }
            self.total_nenergy
        }
    }
    impl DbdConfiguration for CircularConfiguration {
        #[inline]
        fn tcp(&self) -> &TacticalConfigurationParameters { &self.tcp }
        #[inline]
        fn total_nenergy(&self) -> f64 { self.total_nenergy }
        #[inline]
        fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_ {
            let sample_id = sample_id % self.tcp.n_samples();
            self.sequence[sample_id..]
                .iter()
                .chain(self.sequence[..sample_id].iter())
                .take(self.tcp.sample_size().get())
                .copied()
        }
    }
}

/// The circular DBD container
#[must_use]
pub struct DbdCircular<P> {
    /// Annealing temperature tracker
    temperature: AnnealingTemperature,
    /// Energy distance engine
    ed: EnergyDistance<P>,

    /// Main circular config
    configuration: CircularConfiguration,
    /// Best cicular config, if better than main
    configuration_best: Option<CircularConfiguration>,

    /// The switch-candidates to evaluate
    pair: (usize, usize), // k-index
    /// The switch-candidates energy-delta
    total_nenergy_delta: f64,
}
impl<P> DbdCircular<P> {
    /// Returns the optimal configuration
    #[inline]
    pub fn optimal_configuration(&self) -> &CircularConfiguration {
        self.configuration_best
            .as_ref()
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(&self.configuration)
    }
    /// Converts self into the optimal configuration
    #[inline]
    pub fn into_optimal_configuration(self) -> CircularConfiguration {
        self.configuration_best
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(self.configuration)
    }
    /// Returns a reference to the energy distance engine
    #[inline]
    pub fn ed(&self) -> &EnergyDistance<P> { &self.ed }
    /// Returns a reference to the tactical configuration parameters
    #[inline]
    pub fn tcp(&self) -> &TacticalConfigurationParameters { self.configuration.tcp() }

    // CONSTRUCTORS
    /// Constructs a new circular DBD
    ///
    /// # Errors
    /// Returns the full configuration in case of `sample_size` equaling the population size.
    #[expect(
        clippy::unreachable,
        reason = "matrix has rows > 0, one unit must exist in sequence"
    )]
    #[inline]
    pub fn new(
        dbs_options: &DistributionalDesignOptions,
        matrix: P,
        sample_size: NonZeroUsize,
        eps: Epsilon<f64>,
    ) -> Result<Self, CircularConfiguration>
    where
        P: PointSet<Id = usize, Value = f64>,
    {
        let sequence: Box<[usize]> = matrix.ids().collect();
        let annealing_temperature = dbs_options.as_annealing_temperature(eps);
        let ed = EnergyDistance::new(matrix, sample_size);

        let pair = match sequence.as_ref() {
            [a, b, ..] => (*a, *b),
            [a] => (*a, *a),
            _ => unreachable!("empty sequence"),
        };
        let configuration = CircularConfiguration::new(sequence, sample_size, &ed);

        if configuration.tcp().n_samples().get() <= 1 {
            return Err(configuration);
        }

        Ok(Self {
            temperature: annealing_temperature,
            ed,

            configuration,
            configuration_best: None,

            pair,
            total_nenergy_delta: 0.0,
        })
    }
}

impl<P> AnnealingDistributionalDesign for DbdCircular<P>
where
    P: PointSet<Id = usize, Value = f64>,
{
    #[inline]
    fn temperature(&self) -> &AnnealingTemperature { &self.temperature }
    #[inline]
    fn temperature_mut(&mut self) -> &mut AnnealingTemperature { &mut self.temperature }

    #[inline]
    fn draw_units<R>(&mut self, rng: &mut R)
    where
        R: Rand<usize>,
    {
        let n = self.tcp().population_size().get();
        let a = rng.rand_in(0..n);
        let b = rng.rand_in(0..(n - 1));
        self.pair = if a == b { (a, n - 1) } else { (a, b) };
    }
    #[inline]
    fn evaluate_switch(&mut self) -> Option<f64> {
        self.total_nenergy_delta = 0.0;

        // Start off by clearing deltas
        let (k1, k2) = self.pair;
        if k1 == k2 {
            return None;
        }

        let population_size = self.tcp().population_size().get();
        let sample_size = self.tcp().sample_size().get();
        let sample_size_f64 = sample_size
            .to_f64()
            .expect("sample_size should be far below f64 max");
        let d = population_size - sample_size + 1;

        let id1 = self
            .configuration
            .sequence_get(k1)
            .expect("index k1 to exist in the sequence");
        let id2 = self
            .configuration
            .sequence_get(k2)
            .expect("index k2 to exist in the sequence");

        // Units definetly in the overlap should not be counted
        // Other units should be counted relative to their distance to the moving unit
        // Look in the window -sample_size +sample_size around k1
        for k in (k1 + d)..(k1 + d + sample_size * 2 - 1) {
            let k = k % population_size;
            if k == k1 || k == k2 {
                continue;
            }

            let m1: i64 = {
                let abs_dist = k1.abs_diff(k);
                let dist = abs_dist.min(population_size - abs_dist);
                if dist < sample_size {
                    (sample_size - dist)
                        .to_i64()
                        .expect("sample_size - dist to be far below the i64 max")
                } else {
                    0
                }
            };

            let m2: i64 = {
                let abs_dist = k2.abs_diff(k);
                let dist = abs_dist.min(population_size - abs_dist);
                if dist < sample_size {
                    (sample_size - dist)
                        .to_i64()
                        .expect("sample_size - dist to be far below the i64 max")
                } else {
                    0
                }
            };

            if m1 == m2 {
                continue;
            }

            let m_f64 = (2 * (m1 - m2))
                .to_f64()
                .expect("limited by 2 * sample_size, which should be far below f64 max")
                / sample_size_f64;

            let id = self
                .configuration
                .sequence_get(k)
                .expect("k is guaranteed to exist after k % pop_size in beginning of loop");
            let delta = self.ed.relative_distance(id, id1, id2);
            self.total_nenergy_delta += delta * m_f64;
        }

        for k in (k2 + d)..(k2 + d + sample_size * 2 - 1) {
            let k = k % population_size;
            if k == k1 || k == k2 {
                continue;
            }

            {
                // Skip every unit close to k1, as these has already been handled
                let abs_dist = k1.abs_diff(k);
                let dist = abs_dist.min(population_size - abs_dist);
                if dist < sample_size {
                    continue;
                }
            }

            let m_f64: f64 = {
                let abs_dist = k2.abs_diff(k);
                let dist = abs_dist.min(population_size - abs_dist);
                if dist < sample_size {
                    // (2* (sample_size - dist)) / sample_size
                    2.0 * (1.0
                        - dist
                            .to_f64()
                            .expect("limited by sample_size, which should be far below f64 max")
                            / sample_size_f64)
                } else {
                    continue;
                }
            };

            let id = self
                .configuration
                .sequence_get(k)
                .expect("k is guaranteed to exist after k % pop_size in beginning of loop");
            let delta = self.ed.relative_distance(id, id2, id1);
            self.total_nenergy_delta += delta * m_f64;
        }

        Some(self.total_nenergy_delta)
    }
    #[inline]
    fn switch(&mut self) {
        let (k1, k2) = self.pair;
        self.configuration.sequence_mut().swap(k1, k2);
        self.configuration
            .add_nenergy_delta(self.total_nenergy_delta);
    }
    #[inline]
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
