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

//! DBD Utility functions.

use std::num::NonZeroUsize;

use envisim_estimate::spatial_balance::SpatialBalanceError;
use thiserror::Error;

/// DBD errors.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum DbdError {
    /// Error derived from [`SpatialBalanceError`].
    #[error("SamplingBalanceError: {0}")]
    SpatialBalance(#[from] SpatialBalanceError),
    /// Anneling temp below zero.
    #[error("annealing temperature is below 0")]
    AnnealingTemperatureBelowZero,
    /// Anneling rate invalid.
    #[error("annealing rate outside [0, 1)")]
    AnnealingRateInvalid,
    /// Incorrect evaluator interval.
    #[error("evaluator interval invalid")]
    EvaluatorIntervalInvalid,
    /// Sample size is zero.
    #[error("sample size must be positive")]
    SampleSizeZero,
}
impl DbdError {
    /// # Errors
    /// Errors if annealing temp is negative.
    #[inline]
    pub fn check_temp(temp: f64) -> Result<(), Self> {
        if 0.0 <= temp {
            return Ok(());
        }
        Err(Self::AnnealingTemperatureBelowZero)
    }
    /// # Errors
    /// Errors if annealing rate is outside unit interval.
    #[inline]
    pub fn check_rate(rate: f64) -> Result<(), Self> {
        if (0.0..1.0).contains(&rate) {
            return Ok(());
        }
        Err(Self::AnnealingRateInvalid)
    }
    /// # Errors
    /// Errors if evaluation interval is incorrect.
    #[inline]
    pub fn check_eval_interval(to: NonZeroUsize, by: NonZeroUsize) -> Result<(), Self> {
        if by <= to {
            return Ok(());
        }
        Err(Self::EvaluatorIntervalInvalid)
    }
    /// # Errors
    /// If sample size is zero.
    #[inline]
    pub fn check_sample_size(nn: usize) -> Result<NonZeroUsize, Self> {
        NonZeroUsize::new(nn).ok_or(Self::SampleSizeZero)
    }
}
