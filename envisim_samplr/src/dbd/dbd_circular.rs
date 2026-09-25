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

use envisim_estimate::spatial_balance::EnergyDistance;
use envisim_utils::random::Rand;
use envisim_utils::utils::{
    DataView,
    Epsilon,
    PointSet,
};
use num_traits::ToPrimitive;

use super::DistributionalDesignOptions;
use super::annealing::{
    AnnealingDistributionalDesign,
    AnnealingTemperature,
};
pub use super::dbd_circular_config::*;

/// The circular DBD container.
#[must_use]
#[derive(Debug)]
pub struct DbdCircular<PH, P>
where
    PH: DataView<Value = f64>,
    P: PointSet<Id = PH::Id, Value = f64>,
{
    /// Annealing temperature tracker.
    temperature: AnnealingTemperature,
    /// Energy distance engine.
    ed: EnergyDistance<PH, P>,

    /// Main circular config.
    configuration: CircularConfiguration<PH::Id>,
    /// Best cicular config, if better than main.
    configuration_best: Option<CircularConfiguration<PH::Id>>,

    /// The switch-candidates to evaluate.
    pair: (usize, usize), // k-index
    /// The switch-candidates energy-delta.
    total_energy_n_delta: f64,
}
impl<PH, P> DbdCircular<PH, P>
where
    PH: DataView<Value = f64>,
    P: PointSet<Id = PH::Id, Value = f64>,
{
    /// Returns the optimal configuration.
    #[inline]
    pub fn optimal_configuration(&self) -> &CircularConfiguration<PH::Id> {
        self.configuration_best
            .as_ref()
            .filter(|best| best.total_energy_n() <= self.configuration.total_energy_n())
            .unwrap_or(&self.configuration)
    }
    /// Converts self into the optimal configuration.
    #[inline]
    pub fn into_optimal_configuration(self) -> CircularConfiguration<PH::Id> {
        self.configuration_best
            .filter(|best| best.total_energy_n() <= self.configuration.total_energy_n())
            .unwrap_or(self.configuration)
    }
    /// Returns a reference to the energy distance engine.
    #[inline]
    pub fn ed(&self) -> &EnergyDistance<PH, P> { &self.ed }
    /// Returns a reference to the tactical configuration parameters.
    #[inline]
    pub fn tcp(&self) -> &TacticalConfigurationParameters { self.configuration.tcp() }

    // CONSTRUCTORS
    /// Constructs a new circular DBD.
    ///
    /// # Errors
    /// Returns the full configuration in case of `sample_size` equaling the population size.
    #[inline]
    pub fn new(
        ed: EnergyDistance<PH, P>,
        dbs_options: &DistributionalDesignOptions,
        eps: Epsilon<f64>,
    ) -> Result<Self, CircularConfiguration<PH::Id>> {
        let sequence: Box<[_]> = ed.phis().ids().collect();
        let annealing_temperature = dbs_options.as_annealing_temperature(eps);

        let pair = if sequence.len() == 1 { (0, 0) } else { (0, 1) };
        let configuration = CircularConfiguration::new(sequence, &ed);

        if configuration.tcp().n_samples().get() <= 1 {
            return Err(configuration);
        }

        Ok(Self {
            temperature: annealing_temperature,
            ed,

            configuration,
            configuration_best: None,

            pair,
            total_energy_n_delta: 0.0,
        })
    }
}

impl<PH, P> AnnealingDistributionalDesign for DbdCircular<PH, P>
where
    PH: DataView<Value = f64>,
    P: PointSet<Id = PH::Id, Value = f64>,
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
        self.total_energy_n_delta = 0.0;

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
            let delta = self
                .ed
                .distance_difference(id, id1, id2)
                .expect("ids exist");
            self.total_energy_n_delta += delta * m_f64;
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
            let delta = self
                .ed
                .distance_difference(id, id2, id1)
                .expect("ids exist");
            self.total_energy_n_delta += delta * m_f64;
        }

        Some(self.total_energy_n_delta)
    }
    #[inline]
    fn switch(&mut self) {
        let (k1, k2) = self.pair;
        self.configuration.sequence_mut().swap(k1, k2);
        self.configuration
            .add_energy_n_delta(self.total_energy_n_delta);
    }
    #[inline]
    fn set_optimal_configuration(&mut self) {
        // Only replace if conf_best is None, or if we're better than conf_best
        self.configuration_best = match &self.configuration_best {
            // If there is a best, check if the current configuration is better
            Some(best) if self.configuration.total_energy_n() < best.total_energy_n() => {
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
