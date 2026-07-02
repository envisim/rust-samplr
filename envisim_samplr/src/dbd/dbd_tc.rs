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

//! Distributionally balanced designs using tactical configurations

use std::iter::repeat_n;
use std::num::NonZeroUsize;

pub use config::*;
use envisim_utils::random::Rand;
use envisim_utils::sampling_options::{
    Epsilon,
    EqualProbabilityOptions,
    SamplingOptions,
    SamplingOptionsRng,
    UnequalProbabilityOptions,
};
use envisim_utils::spatial::PointSet;

use super::DistributionalDesignOptions;
use super::annealing::{
    AnnealingDistributionalDesign,
    AnnealingTemperature,
};
use super::energy_distance::EnergyDistance;
use crate::pivotal_method::LocalPivotalSampling;
use crate::utils::shuffled_indices;

mod config {
    //! Tactical configuration DBD config

    use envisim_utils::matrix::{
        Dimensions,
        MatrixBase,
    };
    use envisim_utils::spatial::PointSet;

    use crate::dbd::energy_distance::EnergyDistance;
    pub use crate::dbd::tc_parameters::{
        DbdConfiguration,
        TacticalConfigurationParameters,
    };

    /// Storage for the buckets used in a Tactical Configuration
    pub type TcBuckets = MatrixBase<Box<[usize]>>;

    #[must_use]
    #[derive(Clone, Debug)]
    pub struct TacticalConfiguration {
        /// Internal storage of sequence (`sample_size` * `n_samples` matrix)
        // An n * M matrix
        buckets: TcBuckets,
        /// Total energy multiplied by sample size
        total_nenergy: f64,
        /// Tactical configuration parameters
        tcp: TacticalConfigurationParameters,
    }
    impl TacticalConfiguration {
        /// Construct a new circular configuration from a set of indices and tactical configuration
        /// parameters
        ///
        /// # Panics
        /// Panics if the sequence is empty or otherwise incorrect in size. Must be
        /// `n_samples * sample_size`.
        #[inline]
        pub fn new<P>(
            buckets: TcBuckets,
            tcp: TacticalConfigurationParameters,
            ed: &EnergyDistance<P>,
        ) -> Self
        where
            P: PointSet<Id = usize, Value = f64>,
        {
            assert_eq!(
                buckets.dims(),
                (tcp.sample_size(), tcp.n_samples()).into(),
                "invalid sequence length"
            );

            let mut cc = Self {
                buckets,
                total_nenergy: 0.0,
                tcp,
            };

            let _total_energy = cc.reset_total_nenergy(ed);
            cc
        }
        /// Returns a reference to the internal storage
        #[inline]
        pub fn buckets(&self) -> &TcBuckets { &self.buckets }
        /// Consumes `self` and returns the internal storage
        #[inline]
        pub fn into_buckets(self) -> TcBuckets { self.buckets }
        /// Returns a mutable reference to the internal storage
        #[inline]
        pub fn buckets_mut(&mut self) -> &mut TcBuckets { &mut self.buckets }
        /// Add a delta to the nenergy
        #[must_use]
        #[inline]
        pub(crate) fn add_nenergy_delta(&mut self, delta: f64) -> f64 {
            self.total_nenergy += delta;
            self.total_nenergy
        }
        /// Reset the total nenergy
        #[must_use]
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
    impl DbdConfiguration for TacticalConfiguration {
        #[inline]
        fn tcp(&self) -> &TacticalConfigurationParameters { &self.tcp }
        #[inline]
        fn total_nenergy(&self) -> f64 { self.total_nenergy }
        /// # Panics
        /// Panics if `sample_id` is oob.
        #[inline]
        fn sample(&self, sample_id: usize) -> impl Iterator<Item = usize> + Clone + '_ {
            self.buckets
                .col_iter(sample_id)
                .expect("valid sample id")
                .copied()
        }
    }
}

/// The tactical configuration DBD container
#[must_use]
#[derive(Clone, Debug)]
pub struct DbdTacticalConfiguration<P> {
    /// Annealing temperature tracker
    temperature: AnnealingTemperature,
    /// Energy distance engine
    ed: EnergyDistance<P>,

    /// Main circular config
    configuration: TacticalConfiguration,
    /// Best cicular config, if better than main
    configuration_best: Option<TacticalConfiguration>,

    /// The switch-candidates to evaluate
    pair: ((usize, usize), (usize, usize)), // bucket index, k-index within bucket
    /// The switch-candidates energy-delta
    total_nenergy_delta: Option<f64>,
}

impl<P> DbdTacticalConfiguration<P> {
    /// Returns the optimal configuration
    #[inline]
    pub fn optimal_configuration(&self) -> &TacticalConfiguration {
        self.configuration_best
            .as_ref()
            .filter(|best| best.total_nenergy() <= self.configuration.total_nenergy())
            .unwrap_or(&self.configuration)
    }
    /// Converts self into the optimal configuration
    #[inline]
    pub fn into_optimal_configuration(self) -> TacticalConfiguration {
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
    /// Constructs a new tactical configuration DBD
    ///
    /// # Errors
    /// Returns the full configuration in case of `sample_size` equaling the population size.
    #[expect(clippy::panic_in_result_fn, reason = "panic implies bug")]
    #[inline]
    pub fn new<R>(
        rng: &mut R,
        dbs_options: &DistributionalDesignOptions,
        matrix: P,
        sample_size: NonZeroUsize,
        eps: Epsilon<f64>,
    ) -> Result<Self, TacticalConfiguration>
    where
        P: PointSet<Id = usize, Value = f64>,
        R: SamplingOptionsRng<EqualProbabilityOptions>,
    {
        let population_size = matrix.len();
        let sample_size = sample_size.min(population_size);
        let tcp = TacticalConfigurationParameters::new_minimal(population_size, sample_size);

        let mut buckets = TcBuckets::from_value(0, (sample_size, tcp.n_samples()));

        if dbs_options.spatial_initialization() {
            // Initial budget
            let mut b = vec![tcp.n_repeats().get(); tcp.population_size().get()];
            // Offset to samples

            for k in 0..tcp.n_samples().get() {
                // Construct lpm opts
                let p_max = NonZeroUsize::new(tcp.n_samples().get() - k).expect("k < n_samples");
                let p_spec = UnequalProbabilityOptions::<usize>::new_int((&b).into(), p_max)
                    // let p_spec = UnequalProbabilityOptions::new_int(b.into(), p_max)
                    .expect("b to be non-empty and limited by p_max");
                let lpm_opts =
                    // SamplingOptions::<UnequalProbabilityOptions<usize>>::with_spec(p_spec)
                    SamplingOptions::with_spec(p_spec)
                        .set_spreading(&matrix)
                        .expect("matrix to match population size");
                let s = lpm_opts.lpm_2(rng);
                assert_eq!(
                    s.len(),
                    tcp.sample_size().get(),
                    "lpm sample must have the length of the sample_size"
                );

                // For each unit in the sample, reduce the budget
                for (c, id) in buckets
                    .col_iter_mut(k)
                    .expect("column to exist")
                    .zip(s.into_iter())
                {
                    *c = id;
                    b[id] -= 1;
                }
            }
        } else {
            // Add random sequence in next version
            let initial_sequence = shuffled_indices(rng, tcp.population_size());
            // Iterate over ther sequence, repeating each element n_repeats times
            let mut sequence_iter = initial_sequence
                .into_iter()
                .flat_map(|id| repeat_n(id, tcp.n_repeats().get()));
            // Iterate over the matrix row-wise
            for row in 0..sample_size.get() {
                for c in buckets.row_iter_mut(row).expect("row to exist") {
                    *c = sequence_iter.next().expect("sequence to have a unit");
                }
            }
        };

        let annealing_temperature = dbs_options.as_annealing_temperature(eps);
        let ed = EnergyDistance::new(matrix, sample_size);

        let pair = ((0, 0), (1, 0));
        let configuration = TacticalConfiguration::new(buckets, tcp, &ed);

        if configuration.tcp().n_samples().get() <= 1 {
            return Err(configuration);
        }

        Ok(Self {
            temperature: annealing_temperature,
            ed,

            configuration,
            configuration_best: None,

            pair,
            total_nenergy_delta: None,
        })
    }
}

impl<P> AnnealingDistributionalDesign for DbdTacticalConfiguration<P>
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
        let sample_size = self.tcp().sample_size().get();
        let n_samples = self.tcp().n_samples().get();

        loop {
            let a_unit = (rng.rand_in(0..n_samples), rng.rand_in(0..sample_size));
            let mut b_unit = (rng.rand_in(0..(n_samples - 1)), rng.rand_in(0..sample_size));

            if a_unit.0 == b_unit.0 {
                b_unit.0 = n_samples - 1;
            }

            let a_id = self
                .configuration
                .buckets()
                .get((a_unit.1, a_unit.0))
                .copied();
            let b_id = self
                .configuration
                .buckets()
                .get((b_unit.1, b_unit.0))
                .copied();
            if a_id != b_id {
                self.pair = (a_unit, b_unit);
                self.total_nenergy_delta = None;
                return;
            }
        }
    }
    #[inline]
    fn evaluate_switch(&mut self) -> Option<f64> {
        self.total_nenergy_delta = None;

        let ((gr1, k1), (gr2, k2)) = self.pair;
        let id1 = *self
            .configuration
            .buckets()
            .get((k1, gr1))
            .expect("first unit to be valid");
        let id2 = *self
            .configuration
            .buckets()
            .get((k2, gr2))
            .expect("second unit to be valid");

        let delta = self.ed.delta(self.configuration.sample(gr1), id2, id1)?
            + self.ed.delta(self.configuration.sample(gr2), id1, id2)?;

        self.total_nenergy_delta = Some(delta);
        self.total_nenergy_delta
    }
    #[inline]
    fn switch(&mut self) {
        let Some(delta) = self.total_nenergy_delta else {
            return;
        };

        self.configuration.buckets_mut().swap(
            (self.pair.0.1, self.pair.0.0),
            (self.pair.1.1, self.pair.1.0),
        );
        let _total_energy = self.configuration.add_nenergy_delta(delta);
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
