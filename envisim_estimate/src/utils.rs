// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
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

use std::borrow::Borrow;

use envisim_utils::probabilities::FloatProbabilities;

use crate::error::{
    EstimationError,
    EstimationResult,
};

/// Calculates the `y / pi` quotient
///
/// # Errors
/// Returns an error if the `pi` is invalid
#[inline]
pub fn ypi_quotient<Y, PI>((y, pi): (Y, PI)) -> EstimationResult<f64>
where
    Y: Borrow<f64>,
    PI: Borrow<f64>,
{
    let y = *y.borrow();
    let p = *pi.borrow();
    FloatProbabilities::is_prob(p)
        .then(|| y / p)
        .ok_or(EstimationError::InvalidProbability)
}

/// Calculates the `y / pi` quotient
///
/// # Errors
/// Returns an error if any `pi` is invalid
#[inline]
pub fn ypi_iter_to_vec<I, Y, PI>(yp_iter: I) -> EstimationResult<Vec<f64>>
where
    I: Iterator<Item = (Y, PI)>,
    Y: Borrow<f64>,
    PI: Borrow<f64>,
{
    yp_iter.map(ypi_quotient).collect()
}

/// Calculates the `y / mu` quotient
///
/// # Errors
/// Returns an error if the `mu` is invalid
#[inline]
pub fn ymui_quotient<Y, MU, INC>((y, mu, inc): (Y, MU, INC)) -> EstimationResult<f64>
where
    Y: Borrow<f64>,
    MU: Borrow<f64>,
    INC: Borrow<f64>,
{
    let y = *y.borrow();
    let mu = *mu.borrow();
    let inc = *inc.borrow();

    if mu < 0.0 {
        Err(EstimationError::InvalidExpectedNumberOfInclusions)
    } else if inc < 0.0 {
        Err(EstimationError::InvalidNumberOfInclusions)
    } else {
        Ok(y / mu * inc)
    }
}

/// Calculates the `y / mu * inc` quotient
///
/// # Errors
/// Returns an error if any `mu` or `inc` is invalid
#[inline]
pub fn ymui_iter_to_vec<I, Y, MU, INC>(ymui_iter: I) -> EstimationResult<Vec<f64>>
where
    I: Iterator<Item = (Y, MU, INC)>,
    Y: Borrow<f64>,
    MU: Borrow<f64>,
    INC: Borrow<f64>,
{
    ymui_iter.map(ymui_quotient).collect()
}

/// Zips three iterables
#[inline]
pub fn zip3<AI, A, BI, B, CI, C>(a: AI, b: BI, c: CI) -> impl Iterator<Item = (A, B, C)>
where
    AI: IntoIterator<Item = A>,
    BI: IntoIterator<Item = B>,
    CI: IntoIterator<Item = C>,
{
    let a = a.into_iter();
    let b = b.into_iter();
    let c = c.into_iter();
    a.zip(b).zip(c).map(|((x, y), z)| (x, y, z))
}
