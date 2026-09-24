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

//! Tactical configuration DBD config

use envisim_utils::matrix::{
    Dimensions,
    MatrixBase,
};
use envisim_utils::utils::PointSet;

use crate::dbd::energy_distance::EnergyDistance;
pub use crate::dbd::tc_parameters::{
    DbdConfiguration,
    TacticalConfigurationParameters,
};

/// Storage for the buckets used in a Tactical Configuration
pub type TcBuckets = MatrixBase<Box<[usize]>>;

/// Defines a tactical configuration for DBD
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
