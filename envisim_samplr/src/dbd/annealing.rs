// Copyright (C) 2026 Wilmer Prentius
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

//! Methods for annelaing based dbd

use std::num::NonZeroUsize;

use envisim_utils::random::RandomNumberGenerator;

/// Contains the annealing temperature tracker
#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct AnnealingTemperature {
    /// Annealing temperature
    temperature: f64,
    /// Annealing cooling rate
    cooling_rate: f64,
    /// Epsilon value for float comparisons
    eps: f64,
    /// Track number of divergences
    divergence_count: usize,
}
impl AnnealingTemperature {
    /// Constructor
    #[inline]
    pub fn new(temperature: f64, cooling_rate: f64, eps: f64) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        assert!(
            0.0 < cooling_rate && cooling_rate < 1.0,
            "cooling_rate must be in (0.0, 1.0)"
        );
        assert!((0.0..1.0).contains(&eps), "eps must be in [0.0, 1.0)");

        Self {
            temperature,
            cooling_rate,
            eps,
            divergence_count: 0,
        }
    }
    // fn temperature(&self) -> f64 { self.temperature }
    /// Cools the temperature according to the cooling rate
    #[inline]
    fn cool(&mut self) { self.temperature *= self.cooling_rate }
    /// Adds to the divergence counter
    #[inline]
    pub fn increment_divergence(&mut self) { self.divergence_count += 1; }
    /// Returns `true` if temperature is above epsilon
    #[must_use]
    #[inline]
    fn is_positive(&self) -> bool { self.temperature > self.eps }
    /// Returns `true` according to the probabilistic annealing decision
    #[must_use]
    #[inline]
    fn accept_change<R>(&self, rng: &mut R, change: f64) -> bool
    where
        R: RandomNumberGenerator,
    {
        if !self.is_positive() {
            return false;
        }
        let u = rng.rf64();
        let v = (-change / self.temperature).exp();
        u < v
    }
}
impl Default for AnnealingTemperature {
    #[inline]
    fn default() -> Self {
        Self {
            temperature: 0.1,
            cooling_rate: 0.999,
            eps: 1e-12,
            divergence_count: 0,
        }
    }
}

/// Defines an annealing-based dbd
pub trait AnnealingDistributionalDesign {
    /// Returns a reference to the `AnnealingTemperature` object
    fn temperature(&self) -> &AnnealingTemperature;
    /// Returns a mutable reference to the `AnnealingTemperature` object
    fn temperature_mut(&mut self) -> &mut AnnealingTemperature;
    /// Draws the random units
    fn draw_units<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator;
    /// Evaluate the effect of a switch
    #[must_use]
    fn evaluate_switch(&mut self) -> Option<f64>;
    /// Perform a switch
    fn switch(&mut self);
    /// Sets the optimal configuration to the
    fn set_optimal_configuration(&mut self);
    /// Run the optimiziation algorithm for a set number of iterations
    #[inline]
    fn run<R>(&mut self, rng: &mut R, iterations: NonZeroUsize)
    where
        R: RandomNumberGenerator,
    {
        for _ in 0..iterations.get() {
            self.draw_units(rng);
            let Some(delta) = self.evaluate_switch() else {
                continue;
            };

            if delta < 0.0 {
                self.switch();
            } else if self.temperature().accept_change(rng, delta) {
                self.set_optimal_configuration();
                self.switch();
            }

            self.temperature_mut().cool();
        }
    }
}
