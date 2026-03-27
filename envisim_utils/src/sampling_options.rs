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

use std::borrow::Cow;
use std::num::NonZeroUsize;

use crate::kd_tree::{
    FindSplit,
    midpoint_slide,
};
use crate::matrix::Matrix;
use crate::probabilities::{
    ExactProbabilities,
    FloatProbabilities,
    ProbabilityStore,
};
use crate::sample_controller::{
    BasicSampleController,
    SpreadingSampleController,
};
use crate::utils::{
    f64_to_usize,
    usize_to_f64,
};

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

impl std::error::Error for SamplingOptionsError {}

impl std::fmt::Display for SamplingOptionsError {
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
            MissingBalancing => write!(f, "no spreading options provided"),
        }
    }
}

// Probability specification
pub trait ProbabilitySpec {
    type Native: ProbabilityStore;
    fn population_size(&self) -> usize;
    fn population_size_f64(&self) -> f64 { usize_to_f64(self.population_size()) }
    fn sample_size(&self) -> usize;
    fn sample_size_f64(&self) -> f64 { usize_to_f64(self.sample_size()) }
    /// Returns probabilities as f64 slice
    fn as_f64_slice(&self) -> Cow<'_, [f64]>;
    /// Returns probabilities as f64 slice
    fn as_equal(&self) -> Option<ProbabilitySpecEqual> { None }

    fn to_native(&self, eps: f64) -> Self::Native;
    fn to_float(&self, eps: f64) -> FloatProbabilities;
}

#[derive(Clone, Copy, Debug)]
pub struct ProbabilitySpecEqual {
    population_size: usize,
    sample_size: usize,
}
impl From<(usize, usize)> for ProbabilitySpecEqual {
    fn from((population_size, sample_size): (usize, usize)) -> Self {
        ProbabilitySpecEqual {
            population_size,
            sample_size,
        }
    }
}
impl ProbabilitySpecEqual {
    pub fn as_f64(&self) -> f64 { self.sample_size_f64() / self.population_size_f64() }
}
impl ProbabilitySpec for ProbabilitySpecEqual {
    type Native = ExactProbabilities;
    fn population_size(&self) -> usize { self.population_size }
    fn sample_size(&self) -> usize { self.sample_size }
    fn as_f64_slice(&self) -> Cow<'_, [f64]> {
        let p = self.sample_size_f64() / self.population_size_f64();
        Cow::Owned(vec![p; self.population_size])
    }
    fn as_equal(&self) -> Option<ProbabilitySpecEqual> { Some(*self) }
    fn to_native(&self, _eps: f64) -> Self::Native { ExactProbabilities::new_equal(*self) }
    fn to_float(&self, eps: f64) -> FloatProbabilities {
        FloatProbabilities::new(self.as_f64_slice().into_owned(), eps)
    }
}

#[derive(Clone, Debug)]
pub struct ProbabilitySpecUnequal<'a> {
    data: Cow<'a, [f64]>,
}
impl<'a> ProbabilitySpec for ProbabilitySpecUnequal<'a> {
    type Native = FloatProbabilities;
    fn population_size(&self) -> usize { self.data.len() }
    fn sample_size_f64(&self) -> f64 { self.data.iter().sum::<f64>() }
    fn sample_size(&self) -> usize { f64_to_usize(self.sample_size_f64().round()) }
    fn as_f64_slice(&self) -> Cow<'_, [f64]> { Cow::Borrowed(&self.data) }
    fn to_native(&self, eps: f64) -> Self::Native {
        FloatProbabilities::new(self.data.clone().into_owned(), eps)
    }
    fn to_float(&self, eps: f64) -> FloatProbabilities { Self::to_native(self, eps) }
}

// Sub-options
#[derive(Clone, Debug)]
pub struct SpreadingOptions<'a> {
    data: Matrix<'a>,
    bucket_size: NonZeroUsize,
    split_method: FindSplit,
}
impl<'a> SpreadingOptions<'a> {
    pub fn new(data: Matrix<'a>) -> SamplingOptionsResult<Self> {
        if data.nrow() == 0 {
            return Err(SamplingOptionsError::InvalidSpreading);
        }

        let mut opts = Self {
            data,
            bucket_size: NonZeroUsize::new(40).unwrap(),
            split_method: midpoint_slide,
        };
        opts.estimate_bucket_size();
        Ok(opts)
    }

    pub fn data(&self) -> &Matrix<'a> { &self.data }
    pub fn bucket_size(&self) -> NonZeroUsize { self.bucket_size }
    pub fn split_method(&self) -> FindSplit { self.split_method }
    pub fn set_bucket_size(mut self, size: usize) -> SamplingOptionsResult<Self> {
        self.bucket_size =
            NonZeroUsize::new(size).ok_or(SamplingOptionsError::InvalidBucketSize)?;
        Ok(self)
    }
    pub fn set_split_method(mut self, method: FindSplit) -> Self {
        self.split_method = method;
        self
    }
    fn estimate_bucket_size(&mut self) {
        let len = self.data.nrow();
        let size = match len {
            0..=100 => 10,
            101..=400 => len / 10,
            _ => 40,
        };
        self.bucket_size = NonZeroUsize::new(size).unwrap();
    }
}
#[derive(Clone, Debug)]
pub struct BalancingOptions<'a> {
    data: Matrix<'a>,
}
impl<'a> BalancingOptions<'a> {
    pub fn new(data: Matrix<'a>) -> SamplingOptionsResult<Self> {
        if data.nrow() == 0 {
            return Err(SamplingOptionsError::InvalidBalancing);
        }
        Ok(Self { data })
    }

    pub fn data(&self) -> &Matrix<'a> { &self.data }
}

#[derive(Clone, Debug)]
pub struct CoordinationOptions<'a> {
    data: Cow<'a, [f64]>,
}

impl<'a> CoordinationOptions<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        Self {
            data: Cow::Borrowed(data),
        }
    }

    pub fn data(&self) -> &[f64] { &self.data }
}

// Main SamplingOptions
#[derive(Clone, Debug)]
pub struct SamplingOptions<'a, PS: ProbabilitySpec> {
    probabilities: PS,
    eps: f64,
    max_iterations: NonZeroUsize,
    coordination: Option<CoordinationOptions<'a>>,

    spreading: Option<SpreadingOptions<'a>>,
    balancing: Option<BalancingOptions<'a>>,
}
impl<'a, PS: ProbabilitySpec> SamplingOptions<'a, PS> {
    // ACCESSORS
    pub fn probabilities(&'a self) -> &PS { &self.probabilities }
    pub fn population_size(&'a self) -> usize { self.probabilities().population_size() }
    pub fn sample_size(&'a self) -> usize { self.probabilities().sample_size() }
    pub fn eps(&self) -> f64 { self.eps }
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
    pub fn coordination(&self) -> Option<&CoordinationOptions<'a>> { self.coordination.as_ref() }
    pub fn spreading(&'a self) -> Option<&SpreadingOptions<'a>> { self.spreading.as_ref() }
    pub fn get_spreading(&'a self) -> SamplingOptionsResult<&SpreadingOptions<'a>> {
        self.spreading()
            .ok_or(SamplingOptionsError::MissingSpreading)
    }
    pub fn balancing(&'a self) -> Option<&BalancingOptions<'a>> { self.balancing.as_ref() }
    pub fn get_balancing(&'a self) -> SamplingOptionsResult<&BalancingOptions<'a>> {
        self.balancing()
            .ok_or(SamplingOptionsError::MissingBalancing)
    }

    // SETTERS
    pub fn set_eps(mut self, eps: f64) -> SamplingOptionsResult<Self> {
        if !(0.0..1.0).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(self)
    }
    pub fn set_max_iterations(mut self, max: NonZeroUsize) -> SamplingOptionsResult<Self> {
        self.max_iterations = max;
        Ok(self)
    }
    pub fn set_coordination(
        mut self,
        random_values: &'a [f64],
    ) -> Result<Self, SamplingOptionsError> {
        if random_values.len() < self.population_size() {
            return Err(SamplingOptionsError::InvalidRandomValues);
        }
        self.coordination = Some(CoordinationOptions::new(random_values));
        Ok(self)
    }

    pub fn set_spreading(mut self, data: Matrix<'a>) -> SamplingOptionsResult<Self> {
        if data.nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }

        self.spreading = Some(SpreadingOptions::new(data)?);
        Ok(self)
    }
    pub fn set_spreading_opts(
        mut self,
        spreading: SpreadingOptions<'a>,
    ) -> SamplingOptionsResult<Self> {
        if spreading.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }

        self.spreading = Some(spreading);
        Ok(self)
    }
    pub fn set_balancing(mut self, data: Matrix<'a>) -> SamplingOptionsResult<Self> {
        if data.nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }

        self.balancing = Some(BalancingOptions::new(data)?);
        Ok(self)
    }
    pub fn set_balancing_opts(
        mut self,
        balancing: BalancingOptions<'a>,
    ) -> SamplingOptionsResult<Self> {
        if balancing.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }

        self.balancing = Some(balancing);
        Ok(self)
    }

    // BUILDERS
    pub fn to_probabilities(&self) -> PS::Native { self.probabilities.to_native(self.eps) }
    pub fn to_probabilities_float(&self) -> FloatProbabilities {
        self.probabilities.to_float(self.eps)
    }
    pub fn to_controller(&self) -> BasicSampleController<PS::Native> {
        let probs = self.to_probabilities();
        BasicSampleController::new(probs)
    }
    pub fn to_controller_float(&self) -> BasicSampleController<FloatProbabilities> {
        let probs = self.to_probabilities_float();
        BasicSampleController::new(probs)
    }
    pub fn to_spreading_controller(
        &'a self,
    ) -> SamplingOptionsResult<SpreadingSampleController<'a, PS::Native>> {
        let controller = self.to_controller();
        let spreading = self.get_spreading()?;
        SpreadingSampleController::new(controller, spreading)
    }
    pub fn to_spreading_controller_float(
        &'a self,
    ) -> SamplingOptionsResult<SpreadingSampleController<'a, FloatProbabilities>> {
        let controller = self.to_controller_float();
        let spreading = self.get_spreading()?;
        SpreadingSampleController::new(controller, spreading)
    }
}
impl<'a> TryFrom<&'a [f64]> for SamplingOptions<'a, ProbabilitySpecUnequal<'a>> {
    type Error = SamplingOptionsError;
    fn try_from(probabilities: &'a [f64]) -> Result<Self, Self::Error> { Self::new(probabilities) }
}
impl<'a> TryFrom<Vec<f64>> for SamplingOptions<'a, ProbabilitySpecUnequal<'a>> {
    type Error = SamplingOptionsError;
    fn try_from(probabilities: Vec<f64>) -> Result<Self, Self::Error> {
        Self::new_owned(probabilities)
    }
}
impl<'a> TryFrom<ProbabilitySpecEqual> for SamplingOptions<'a, ProbabilitySpecEqual> {
    type Error = SamplingOptionsError;
    fn try_from(spec: ProbabilitySpecEqual) -> Result<Self, Self::Error> {
        Self::new_equal_spec(spec)
    }
}
impl<'a> TryFrom<(usize, usize)> for SamplingOptions<'a, ProbabilitySpecEqual> {
    type Error = SamplingOptionsError;
    fn try_from((population_size, sample_size): (usize, usize)) -> Result<Self, Self::Error> {
        Self::new_equal(population_size, sample_size)
    }
}
impl<'a> SamplingOptions<'a, ProbabilitySpecUnequal<'a>> {
    pub fn new(probabilities: &'a [f64]) -> SamplingOptionsResult<Self> {
        if probabilities.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        if !probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }

        const EPS: f64 = 1e-9;
        Ok(Self {
            probabilities: ProbabilitySpecUnequal {
                data: Cow::Borrowed(probabilities),
            },
            eps: EPS,
            max_iterations: NonZeroUsize::new(1000).unwrap(),
            coordination: None,
            spreading: None,
            balancing: None,
        })
    }
    pub fn new_owned(probabilities: Vec<f64>) -> Result<Self, SamplingOptionsError> {
        if probabilities.is_empty() {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        if !probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }

        const EPS: f64 = 1e-9;
        Ok(Self {
            probabilities: ProbabilitySpecUnequal {
                data: Cow::Owned(probabilities),
            },
            eps: EPS,
            max_iterations: NonZeroUsize::new(1000).unwrap(),
            coordination: None,
            spreading: None,
            balancing: None,
        })
    }
}
impl<'a> SamplingOptions<'a, ProbabilitySpecEqual> {
    pub fn new_equal_spec(spec: ProbabilitySpecEqual) -> SamplingOptionsResult<Self> {
        if spec.population_size == 0 {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        }
        if spec.sample_size > spec.population_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }

        Ok(Self {
            probabilities: spec,
            eps: 1e-9,
            max_iterations: NonZeroUsize::new(1000).unwrap(),
            coordination: None,
            spreading: None,
            balancing: None,
        })
    }
    pub fn new_equal(population_size: usize, sample_size: usize) -> SamplingOptionsResult<Self> {
        Self::new_equal_spec((population_size, sample_size).into())
    }
}
