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

//! Estimation utility functions

use envisim_utils::probabilities::Probability;
use envisim_utils::utils::Number;

use crate::error::{
    EstimationError,
    EstimationResult,
};

/// Calculates the `y / pi` quotient
///
/// # Errors
/// Returns an error if the `pi` is invalid
#[inline]
pub fn ypi_quotient<Y>((y, pi): (Y, f64)) -> EstimationResult<f64>
where
    Y: Number,
{
    if !Probability::is_real_probability(pi) {
        return Err(EstimationError::InvalidProbability);
    }
    y.to_f64()
        .map(|y| y / pi)
        .ok_or(EstimationError::InvalidAuxiliaries)
}

/// Calculates the `y / mu` quotient
///
/// # Errors
/// Returns an error if the `mu` is invalid
#[inline]
pub fn ymui_quotient<Y, INC>((y, mu, inc): (Y, f64, INC)) -> EstimationResult<f64>
where
    Y: Number,
    INC: Number,
{
    if mu < 0.0 || !mu.is_finite() {
        Err(EstimationError::InvalidExpectedNumberOfInclusions)
    } else if inc < INC::ZERO || !inc.is_finite() {
        Err(EstimationError::InvalidNumberOfInclusions)
    } else {
        let y = y.to_f64().ok_or(EstimationError::InvalidAuxiliaries)?;
        let inc = inc
            .to_f64()
            .ok_or(EstimationError::InvalidExpectedNumberOfInclusions)?;
        Ok(y / mu * inc)
    }
}
