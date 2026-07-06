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

//! Distributionally balanced design options

use envisim_utils::utils::Epsilon;

use super::annealing::AnnealingTemperature;
use crate::{
    SamplingError,
    SamplingResult,
};

/// Design options for distributionally balanced designs
#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct DistributionalDesignOptions {
    /// The starting temperature, must be positive
    annealing_temperature: f64,
    /// The cooling rate of the temperature, must be in (0.0, 1.0)
    annealing_cooling_rate: f64,
    /// If search should start with a spatial initialization
    spatial_initialization: bool,
}
impl Default for DistributionalDesignOptions {
    #[inline]
    fn default() -> Self {
        Self {
            annealing_temperature: 0.1,
            annealing_cooling_rate: 0.999,
            spatial_initialization: false,
        }
    }
}
impl DistributionalDesignOptions {
    /// Returns the initial annealing temperature
    #[must_use]
    #[inline]
    pub fn annealing_temperature(&self) -> f64 { self.annealing_temperature }
    /// Returns the annealing cooling rate
    #[must_use]
    #[inline]
    pub fn annealing_cooling_rate(&self) -> f64 { self.annealing_cooling_rate }
    /// Returns the spatial initialization indicator
    #[must_use]
    #[inline]
    pub fn spatial_initialization(&self) -> bool { self.spatial_initialization }
    /// Constructs a new dbd option with a given initial temperature
    ///
    /// # Errors
    /// Returns an error if `temp` is not positive.
    #[inline]
    pub fn new(temp: f64) -> SamplingResult<Self> {
        let opts = Self::default();
        opts.set_annealing_temperature(temp)
    }
    /// Sets the initial annealing temperature
    ///
    /// # Errors
    /// Returns an error if `temp` is not positive.
    #[inline]
    pub fn set_annealing_temperature(mut self, temp: f64) -> SamplingResult<Self> {
        if temp <= 0.0 {
            return Err(SamplingError::IncorrectAnnealingTemperature);
        }
        self.annealing_temperature = temp;
        Ok(self)
    }
    /// Sets the annealing cooling rate
    ///
    /// # Errors
    /// Returns an error if rate is not contained in $[0.0, 1.0)$
    #[inline]
    pub fn set_annealing_cooling_rate(mut self, rate: f64) -> SamplingResult<Self> {
        if !(0.0..1.0).contains(&rate) {
            return Err(SamplingError::IncorrectAnnealingRate);
        }
        self.annealing_cooling_rate = rate;
        Ok(self)
    }
    /// Sets the spatial initialization indicator
    #[inline]
    pub fn set_spatial_initialization(mut self, spatial_init: bool) -> Self {
        self.spatial_initialization = spatial_init;
        self
    }
    /// Converts the dbd option into an `AnnealingTemperature` object
    #[inline]
    pub(crate) fn as_annealing_temperature(&self, eps: Epsilon<f64>) -> AnnealingTemperature {
        AnnealingTemperature::new(self.annealing_temperature, self.annealing_cooling_rate, eps)
    }
}
