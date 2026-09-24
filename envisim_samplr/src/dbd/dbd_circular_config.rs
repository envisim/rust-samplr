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
    pub fn new<P>(sequence: Box<[usize]>, sample_size: NonZeroUsize, ed: &EnergyDistance<P>) -> Self
    where
        P: PointSet<Id = usize, Value = f64>,
    {
        let population_size = NonZeroUsize::new(sequence.len()).expect("sequence to be non-empty");
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
