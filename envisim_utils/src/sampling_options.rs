// Copyright (C) 2025 Wilmer Prentius.
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
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use crate::kd_tree::{
    FindSplit,
    midpoint_slide,
};
use crate::matrix::Matrix;
use crate::probabilities::Probabilities;
pub use crate::probabilities::{
    ProbabilitiesEqual,
    ProbabilitiesUnequal,
};
use crate::utils::{
    f64_to_usize,
    usize_to_f64,
};

pub struct Enabled;
pub struct Disabled;

#[derive(Clone, Debug)]
#[allow(clippy::exhaustive_enums)]
pub enum ProbabilitySpec<'a> {
    Equal { sample_size: usize },
    Unequal { values: Cow<'a, [f64]> },
}

#[derive(Clone, Debug)]
pub struct IncProbOptions<'a, P = ProbabilitiesUnequal>
where
    P: Probabilities,
{
    spec: ProbabilitySpec<'a>,
    population_size: usize,
    _phantom: PhantomData<P>,
}
impl<'a, P> IncProbOptions<'a, P>
where
    P: Probabilities,
{
    pub fn spec(&self) -> &ProbabilitySpec<'a> { &self.spec }
    pub fn population_size(&self) -> usize { self.population_size }
    pub fn slice(&'a self) -> Cow<'a, [f64]> {
        match self.spec {
            ProbabilitySpec::Equal { sample_size } => {
                let ip = usize_to_f64(sample_size) / usize_to_f64(self.population_size);
                Cow::Owned(vec![ip; self.population_size])
            }
            ProbabilitySpec::Unequal { ref values } => Cow::Borrowed(values),
        }
    }
    pub fn try_slice_equal(&self) -> Option<Vec<usize>> {
        match self.spec {
            ProbabilitySpec::Equal { sample_size } => Some(vec![sample_size; self.population_size]),
            _ => None,
        }
    }
    pub fn sample_size_usize(&self) -> usize {
        match self.spec {
            ProbabilitySpec::Equal { sample_size } => sample_size,
            ProbabilitySpec::Unequal { ref values } => {
                f64_to_usize(values.iter().sum::<f64>().round())
            }
        }
    }
    pub fn sample_size_f64(&self) -> f64 {
        match self.spec {
            ProbabilitySpec::Equal { sample_size } => usize_to_f64(sample_size),
            ProbabilitySpec::Unequal { ref values } => values.iter().sum::<f64>().round(),
        }
    }
}
impl<'a> IncProbOptions<'a, ProbabilitiesEqual> {
    pub fn slice_equal(&self) -> Vec<usize> {
        self.try_slice_equal().expect("guaranteed by type system")
    }
    pub fn sample_size(&self) -> usize { self.sample_size_usize() }
}
impl<'a, P> Default for IncProbOptions<'a, P>
where
    P: Probabilities,
{
    fn default() -> IncProbOptions<'a, P> {
        IncProbOptions::<'a, P> {
            spec: ProbabilitySpec::Equal { sample_size: 0 },
            population_size: 0,
            _phantom: PhantomData,
        }
    }
}
impl<'a> From<&'a [f64]> for IncProbOptions<'a, ProbabilitiesUnequal> {
    fn from(arr: &'a [f64]) -> IncProbOptions<'a, ProbabilitiesUnequal> {
        let population_size = arr.len();
        IncProbOptions::<ProbabilitiesUnequal> {
            spec: ProbabilitySpec::Unequal {
                values: Cow::Borrowed(arr),
            },
            population_size,
            _phantom: PhantomData,
        }
    }
}
impl<'a> From<Vec<f64>> for IncProbOptions<'a, ProbabilitiesUnequal> {
    fn from(arr: Vec<f64>) -> IncProbOptions<'a, ProbabilitiesUnequal> {
        let population_size = arr.len();
        IncProbOptions::<ProbabilitiesUnequal> {
            spec: ProbabilitySpec::Unequal {
                values: Cow::Owned(arr),
            },
            population_size,
            _phantom: PhantomData,
        }
    }
}
impl<'a> From<(usize, usize)> for IncProbOptions<'a, ProbabilitiesEqual> {
    fn from(pair: (usize, usize)) -> IncProbOptions<'a, ProbabilitiesEqual> {
        IncProbOptions::<ProbabilitiesEqual> {
            spec: ProbabilitySpec::Equal {
                sample_size: pair.1,
            },
            population_size: pair.0,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpreadingOptions<'a> {
    data: Matrix<'a>,
    bucket_size: NonZeroUsize,
    split_method: FindSplit,
}

impl<'a> SpreadingOptions<'a> {
    pub fn new(data: Matrix<'a>) -> SpreadingOptions<'a> {
        if data.nrow() == 0 {
            panic!("matrix 'data' has no data (nrows = 0)");
        }

        Self {
            data,
            bucket_size: unsafe { NonZeroUsize::new_unchecked(40) },
            split_method: midpoint_slide,
        }
        .est_bucket_size()
    }

    pub fn data(&self) -> &Matrix<'a> { &self.data }
    pub fn bucket_size(&self) -> NonZeroUsize { self.bucket_size }
    pub fn set_bucket_size(
        mut self,
        bucket_size: impl TryInto<NonZeroUsize>,
    ) -> Result<Self, SamplingOptionsError> {
        let Ok(bs) = bucket_size.try_into() else {
            return Err(SamplingOptionsError::InvalidBucketSize);
        };
        self.bucket_size = bs;
        Ok(self)
    }
    pub fn est_bucket_size(mut self) -> SpreadingOptions<'a> {
        let len = self.data.nrow();
        let bucket_size = match len {
            0usize..=100 => 10usize,
            101usize..=400 => len / 10usize,
            _ => 40usize,
        };
        self.bucket_size = unsafe { NonZeroUsize::new_unchecked(bucket_size) };
        self
    }
    pub fn split_method(&self) -> FindSplit { self.split_method }
    pub fn set_split_method(mut self, split_method: FindSplit) -> Self {
        self.split_method = split_method;
        self
    }
}

impl<'a> From<&'a Matrix<'a>> for SpreadingOptions<'a> {
    fn from(data: &'a Matrix<'a>) -> SpreadingOptions<'a> { Self::new(data.clone_shallow()) }
}
impl<'a> From<Matrix<'a>> for SpreadingOptions<'a> {
    fn from(data: Matrix<'a>) -> SpreadingOptions<'a> { Self::new(data) }
}

#[derive(Clone, Debug)]
pub struct BalancingOptions<'a> {
    data: Matrix<'a>,
}

impl<'a> BalancingOptions<'a> {
    #[inline]
    pub fn new(data: Matrix<'a>) -> Self {
        if data.nrow() == 0 {
            panic!("matrix 'data' has no data (nrows = 0)");
        }

        Self { data }
    }
    pub fn data(&self) -> &Matrix<'a> { &self.data }
}
impl<'a> From<&'a Matrix<'a>> for BalancingOptions<'a> {
    fn from(data: &'a Matrix<'a>) -> Self { Self::new(data.clone_shallow()) }
}
impl<'a> From<Matrix<'a>> for BalancingOptions<'a> {
    fn from(data: Matrix<'a>) -> Self { Self::new(data) }
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
impl<'a> From<&'a [f64]> for CoordinationOptions<'a> {
    fn from(data: &'a [f64]) -> Self { Self::new(data) }
}
impl<'a> From<Vec<f64>> for CoordinationOptions<'a> {
    fn from(data: Vec<f64>) -> Self {
        Self {
            data: Cow::Owned(data),
        }
    }
}

pub struct SpreadingDisabled;
pub struct SpreadingEnabled;
pub struct BalancingDisabled;
pub struct BalancingEnabled;

#[derive(Clone, Debug)]
pub struct SamplingOptions<'a, P = ProbabilitiesUnequal, S = Disabled, B = Disabled>
where
    P: Probabilities,
{
    probabilities: IncProbOptions<'a, P>,

    // Base
    eps: f64,
    max_iterations: NonZeroUsize,

    // Coordinated
    coordination: Option<CoordinationOptions<'a>>,

    // Spatially balanced sampling
    spreading: Option<SpreadingOptions<'a>>,

    // Balanced sampling
    balancing: Option<BalancingOptions<'a>>,

    // Phantom
    _phantom: PhantomData<(S, B)>,
}
impl<'a, P> Default for SamplingOptions<'a, P, Disabled, Disabled>
where
    P: Probabilities,
{
    fn default() -> SamplingOptions<'a, P, Disabled, Disabled> {
        SamplingOptions::<'a, P, Disabled, Disabled> {
            probabilities: Default::default(),
            eps: 1e-12,
            max_iterations: unsafe { NonZeroUsize::new_unchecked(1000) },
            coordination: None,
            spreading: None,
            balancing: None,
            _phantom: PhantomData,
        }
    }
}

impl<'a, P, S, B> SamplingOptions<'a, P, S, B>
where
    P: Probabilities,
{
    pub fn probabilities(&self) -> &IncProbOptions<'a, P> { &self.probabilities }
    pub fn population_size(&self) -> usize { self.probabilities.population_size() }
    pub fn set_probabilities_unequal(
        self,
        probabilities: &'a [f64],
    ) -> Result<SamplingOptions<'a, ProbabilitiesUnequal, S, B>, SamplingOptionsError> {
        let population_size = probabilities.len();
        if population_size == 0 {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        if let Some(ref rv) = self.coordination {
            if rv.data.len() < population_size {
                return Err(SamplingOptionsError::InvalidPopulationSize);
            }
        }
        if let Some(ref spr) = self.spreading {
            if spr.data.nrow() != population_size {
                return Err(SamplingOptionsError::InvalidPopulationSize);
            }
        }
        if let Some(ref bal) = self.balancing {
            if bal.data.nrow() != population_size {
                return Err(SamplingOptionsError::InvalidPopulationSize);
            }
        }

        let p_opts: IncProbOptions<'a, ProbabilitiesUnequal> = probabilities.into();
        let opts = SamplingOptions::<'a, ProbabilitiesUnequal, S, B> {
            probabilities: p_opts,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: self.spreading,
            balancing: self.balancing,
            _phantom: PhantomData,
        };
        Ok(opts)
    }
    pub fn set_probabilities_equal(
        self,
        sample_size: usize,
    ) -> Result<SamplingOptions<'a, ProbabilitiesEqual, S, B>, SamplingOptionsError> {
        let population_size = self.population_size();
        if population_size < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }

        let p_opts: IncProbOptions<'a, ProbabilitiesEqual> = (population_size, sample_size).into();
        let opts = SamplingOptions::<'a, ProbabilitiesEqual, S, B> {
            probabilities: p_opts,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: self.spreading,
            balancing: self.balancing,
            _phantom: PhantomData,
        };
        Ok(opts)
    }
    pub fn eps(&self) -> f64 { self.eps }
    pub fn set_eps(mut self, eps: f64) -> Result<Self, SamplingOptionsError> {
        if !(0.0..1.0).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(self)
    }
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
    pub fn set_max_iterations(
        mut self,
        max_iterations: NonZeroUsize,
    ) -> Result<Self, SamplingOptionsError> {
        self.max_iterations = max_iterations;
        Ok(self)
    }
    pub fn coordination(&self) -> Option<&CoordinationOptions<'a>> { self.coordination.as_ref() }
    pub fn set_coordination(
        mut self,
        random_values: &'a [f64],
    ) -> Result<Self, SamplingOptionsError> {
        if random_values.len() < self.population_size() {
            return Err(SamplingOptionsError::InvalidRandomValues);
        }
        self.coordination = Some(random_values.into());
        Ok(self)
    }
    pub fn set_spreading(
        self,
        spreading: impl Into<SpreadingOptions<'a>>,
    ) -> Result<SamplingOptions<'a, P, Enabled, B>, SamplingOptionsError> {
        let spreading = spreading.into();
        if spreading.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }

        let opts = SamplingOptions::<'a, P, Enabled, B> {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: Some(spreading),
            balancing: self.balancing,
            _phantom: PhantomData,
        };
        Ok(opts)
    }
    pub fn set_balancing(
        self,
        balancing: impl Into<BalancingOptions<'a>>,
    ) -> Result<SamplingOptions<'a, P, S, Enabled>, SamplingOptionsError> {
        let balancing = balancing.into();
        if balancing.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }

        let opts = SamplingOptions::<'a, P, S, Enabled> {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: self.spreading,
            balancing: Some(balancing),
            _phantom: PhantomData,
        };
        Ok(opts)
    }
}
impl<'a, S, B> SamplingOptions<'a, ProbabilitiesEqual, S, B> {
    pub fn sample_size(&self) -> usize { self.probabilities.sample_size() }
}
impl<'a> SamplingOptions<'a, ProbabilitiesUnequal, Disabled, Disabled> {
    pub fn new(
        probabilities: &'a [f64],
    ) -> Result<SamplingOptions<'a, ProbabilitiesUnequal, Disabled, Disabled>, SamplingOptionsError>
    {
        let population_size = probabilities.len();
        if population_size == 0 {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if !probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)) {
            return Err(SamplingOptionsError::InvalidProbability);
        }

        let p_opts: IncProbOptions<'a, ProbabilitiesUnequal> = probabilities.into();
        let opts = SamplingOptions::<'a, ProbabilitiesUnequal> {
            probabilities: p_opts,
            ..Default::default()
        };
        Ok(opts)
    }
}
impl<'a> SamplingOptions<'a, ProbabilitiesEqual, Disabled, Disabled> {
    pub fn new_equal(
        population_size: usize,
        sample_size: usize,
    ) -> Result<SamplingOptions<'a, ProbabilitiesEqual>, SamplingOptionsError> {
        if population_size == 0 {
            return Err(SamplingOptionsError::InvalidPopulationSize);
        } else if population_size < sample_size {
            return Err(SamplingOptionsError::InvalidSampleSize);
        }

        let p_opts: IncProbOptions<'a, ProbabilitiesEqual> = (population_size, sample_size).into();
        let opts = SamplingOptions::<'a, ProbabilitiesEqual> {
            probabilities: p_opts,
            ..Default::default()
        };
        Ok(opts)
    }
}

impl<'a> TryFrom<&'a [f64]> for SamplingOptions<'a, ProbabilitiesUnequal> {
    type Error = SamplingOptionsError;
    fn try_from(
        probabilities: &'a [f64],
    ) -> Result<SamplingOptions<'a, ProbabilitiesUnequal>, Self::Error> {
        Self::new(probabilities)
    }
}
impl<'a> TryFrom<(usize, usize)> for SamplingOptions<'a, ProbabilitiesEqual> {
    type Error = SamplingOptionsError;
    fn try_from(
        (population_size, sample_size): (usize, usize),
    ) -> Result<SamplingOptions<'a, ProbabilitiesEqual>, Self::Error> {
        Self::new_equal(population_size, sample_size)
    }
}

impl<'a, P, B> SamplingOptions<'a, P, Enabled, B>
where
    P: Probabilities,
{
    pub fn spreading(&self) -> &SpreadingOptions<'a> {
        self.spreading.as_ref().expect("Spreading Enabled")
    }
}
impl<'a, P, S> SamplingOptions<'a, P, S, Enabled>
where
    P: Probabilities,
{
    pub fn balancing(&'a self) -> &'a BalancingOptions<'a> {
        self.balancing.as_ref().expect("Balancing Enabled")
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum SamplingOptionsError {
    InvalidEpsilon,
    InvalidPopulationSize,
    InvalidSampleSize,
    InvalidProbability,
    InvalidRandomValues,
    InvalidSpreading,
    InvalidBucketSize,
    InvalidBalancing,
}
impl std::error::Error for SamplingOptionsError {}
impl std::fmt::Display for SamplingOptionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use SamplingOptionsError::*;
        match *self {
            InvalidEpsilon => write!(f, "eps must be in [0.0, 1.0)"),
            InvalidPopulationSize => write!(f, "population is empty"),
            InvalidSampleSize => write!(f, "sample size must be smaller than population size"),
            InvalidProbability => write!(f, "probabilities must be in [0.0, 1.0]"),
            InvalidRandomValues => write!(
                f,
                "size of random values must be at least as large as population size"
            ),
            InvalidSpreading => write!(f, "spreading matrix must be of population size"),
            InvalidBucketSize => write!(f, "bucket size must be at least 1"),
            InvalidBalancing => write!(f, "balancing matrix must be of population size"),
        }
    }
}
