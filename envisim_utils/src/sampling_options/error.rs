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

use std::error::Error;
use std::fmt::Display;

#[non_exhaustive]
#[derive(Debug)]
pub enum SamplingOptionsError {
    InvalidEpsilon,
    InvalidPopulationSize,
    InvalidSample,
    InvalidSampleSize,
    InvalidProbability,
    InvalidRandomValues,
    InvalidSpreading,
    InvalidBucketSize,
    InvalidBalancing,
    MissingSpreading,
    MissingBalancing,
}
pub type SamplingOptionsResult<T> = Result<T, SamplingOptionsError>;

impl Error for SamplingOptionsError {}

impl Display for SamplingOptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use SamplingOptionsError::*;
        match self {
            InvalidEpsilon => write!(f, "eps must be in [0.0, 1.0)"),
            InvalidPopulationSize => write!(f, "population is empty"),
            InvalidSample => write!(f, "sample contains invalid units"),
            InvalidSampleSize => write!(f, "sample size must be smaller than population size"),
            InvalidProbability => write!(f, "probabilities must be in [0.0, 1.0]"),
            InvalidRandomValues => write!(f, "random values length must be >= population size"),
            InvalidSpreading => write!(f, "spreading matrix must have population_size rows"),
            InvalidBucketSize => write!(f, "bucket size must be at least 1"),
            InvalidBalancing => write!(f, "balancing matrix must have population_size rows"),
            MissingSpreading => write!(f, "no spreading options provided"),
            MissingBalancing => write!(f, "no balancing options provided"),
        }
    }
}
