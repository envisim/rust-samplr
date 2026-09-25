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

use envisim_estimate::spatial_balance::EnergyDistance;
use envisim_utils::utils::{
    DataView,
    PointSet,
};

pub use crate::dbd::tc_parameters::{
    DbdConfiguration,
    TacticalConfigurationParameters,
};

/// Stores a circular design configuration
#[must_use]
#[derive(Clone, Debug)]
pub struct CircularConfiguration<ID> {
    /// Internal storage of sequence
    sequence: Box<[ID]>,
    /// Total energy multiplied by sample size
    total_energy_n: f64,
    /// Tactical configuration parameters
    tcp: TacticalConfigurationParameters,
}
impl<ID> CircularConfiguration<ID>
where
    ID: Copy,
{
    /// Construct a new circular configuration from a sequence
    ///
    /// # Panics
    /// Panics if the sequence is empty.
    #[inline]
    pub fn new<PH, P>(sequence: Box<[ID]>, ed: &EnergyDistance<PH, P>) -> Self
    where
        PH: DataView<Id = ID, Value = f64>,
        P: PointSet<Id = ID, Value = f64>,
    {
        let population_size = NonZeroUsize::new(sequence.len()).expect("sequence to be non-empty");
        let sample_size = ed.sample_size().min(population_size);

        let mut cc = Self {
            sequence,
            total_energy_n: 0.0,
            tcp: TacticalConfigurationParameters::new(
                population_size,
                sample_size,
                population_size,
                sample_size,
            ),
        };

        cc.reset_total_energy_n(ed);
        cc
    }
    /// Returns a reference to the sequence store
    #[must_use]
    #[inline]
    pub fn sequence(&self) -> &[ID] { &self.sequence }
    /// Consumes `self` and returns the internal storage
    #[must_use]
    #[inline]
    pub fn into_sequence(self) -> Box<[ID]> { self.sequence }
    /// Returns a mutable reference to the sequence store
    #[must_use]
    #[inline]
    pub fn sequence_mut(&mut self) -> &mut [ID] { &mut self.sequence }
    /// Returns an element from the sequence store
    #[must_use]
    #[inline]
    pub fn sequence_get(&self, k: usize) -> Option<ID>
    where
        ID: Copy,
    {
        self.sequence.get(k).copied()
    }
    /// Add a delta to the nenergy
    pub(crate) fn add_energy_n_delta(&mut self, delta: f64) -> f64 {
        self.total_energy_n += delta;
        self.total_energy_n
    }
    /// Reset the total nenergy
    #[inline]
    fn reset_total_energy_n<PH, P>(&mut self, ed: &EnergyDistance<PH, P>) -> f64
    where
        PH: DataView<Id = ID, Value = f64>,
        P: PointSet<Id = ID, Value = f64>,
    {
        self.total_energy_n = 0.0;
        for i in 0..self.tcp.n_samples().get() {
            self.total_energy_n += self.energy_of_sample_n(ed, i);
        }
        self.total_energy_n
    }
}
impl<ID> DbdConfiguration<ID> for CircularConfiguration<ID>
where
    ID: Copy,
{
    #[inline]
    fn tcp(&self) -> &TacticalConfigurationParameters { &self.tcp }
    #[inline]
    fn total_energy_n(&self) -> f64 { self.total_energy_n }
    #[inline]
    fn sample(&self, sample_id: usize) -> impl Iterator<Item = ID> + Clone + '_ {
        let sample_id = sample_id % self.tcp.n_samples();
        self.sequence
            .iter()
            .skip(sample_id)
            .chain(self.sequence.iter().take(sample_id))
            .take(self.tcp.sample_size().get())
            .copied()
    }
}
