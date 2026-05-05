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

mod balancing_opts;
mod coordination_opts;
mod error;
mod probability_spec;
mod spreading_opts;

use std::num::NonZeroUsize;

pub use balancing_opts::BalancingOptions;
pub use coordination_opts::CoordinationOptions;
pub use error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
pub use probability_spec::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
    ProbabilitySpecUnequal,
};
pub use spreading_opts::SpreadingOptions;

use crate::matrix::MatrixRef;
use crate::number_traits::Number;
use crate::probabilities::FloatProbabilities;
use crate::sample_controller::{
    BasicSampleController,
    SpreadingSampleController,
};
use crate::spatial::PointSet;

#[derive(Clone, Debug)]
pub struct SamplingOptions<'a, PS, SOP = (), M = ()> {
    probabilities: PS,
    eps: f64,
    max_iterations: NonZeroUsize,
    coordination: Option<CoordinationOptions<'a>>,

    spreading: Option<SpreadingOptions<SOP>>,
    balancing: Option<BalancingOptions<'a, M>>,
}

impl<'a, PS, SOP, M> SamplingOptions<'a, PS, SOP, M> {
    // ACCESSORS
    #[inline]
    pub fn probabilities(&self) -> &PS { &self.probabilities }
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize
    where
        PS: ProbabilitySpec,
    {
        self.probabilities().population_size()
    }
    #[inline]
    pub fn sample_size(&self) -> usize
    where
        PS: ProbabilitySpec,
    {
        self.probabilities().sample_size()
    }
    #[inline]
    pub fn eps(&self) -> f64 { self.eps }
    #[inline]
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
    #[inline]
    pub fn coordination(&self) -> Option<&CoordinationOptions<'a>> { self.coordination.as_ref() }
    #[inline]
    pub fn spreading(&self) -> SamplingOptionsResult<&SpreadingOptions<SOP>> {
        self.spreading
            .as_ref()
            .ok_or(SamplingOptionsError::MissingSpreading)
    }
    #[inline]
    pub fn balancing(&self) -> SamplingOptionsResult<&BalancingOptions<'a, M>> {
        self.balancing
            .as_ref()
            .ok_or(SamplingOptionsError::MissingBalancing)
    }

    // SETTERS
    #[inline]
    pub fn set_eps(mut self, eps: f64) -> SamplingOptionsResult<Self> {
        if !(0.0..0.001).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(self)
    }
    #[inline]
    pub fn set_max_iterations(mut self, max: NonZeroUsize) -> SamplingOptionsResult<Self> {
        self.max_iterations = max;
        Ok(self)
    }
    #[inline]
    pub fn set_coordination(
        mut self,
        random_values: &'a [f64],
    ) -> Result<Self, SamplingOptionsError>
    where
        PS: ProbabilitySpec,
    {
        if random_values.len() < self.population_size().get() {
            return Err(SamplingOptionsError::InvalidRandomValues);
        }
        self.coordination = Some(CoordinationOptions::new(random_values)?);
        Ok(self)
    }

    #[inline]
    pub fn set_spreading<NewSOP, N>(
        self,
        data: NewSOP,
    ) -> SamplingOptionsResult<SamplingOptions<'a, PS, NewSOP, M>>
    where
        PS: ProbabilitySpec,
        NewSOP: PointSet<N>,
    {
        if data.size() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: Some(SpreadingOptions::new(data)?),
            balancing: self.balancing,
        })
    }
    #[inline]
    pub fn set_spreading_opts<NewSOP, N>(
        self,
        spreading: SpreadingOptions<NewSOP>,
    ) -> SamplingOptionsResult<SamplingOptions<'a, PS, NewSOP, M>>
    where
        PS: ProbabilitySpec,
        NewSOP: PointSet<N>,
    {
        if spreading.data().size() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: Some(spreading),
            balancing: self.balancing,
        })
    }
    #[inline]
    pub fn set_balancing<NewM>(
        self,
        data: MatrixRef<'a, NewM>,
    ) -> SamplingOptionsResult<SamplingOptions<'a, PS, SOP, NewM>>
    where
        PS: ProbabilitySpec,
    {
        if data.nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: self.spreading,
            balancing: Some(BalancingOptions::new(data)?),
        })
    }
    #[inline]
    pub fn set_balancing_opts<NewM>(
        self,
        balancing: BalancingOptions<'a, NewM>,
    ) -> SamplingOptionsResult<SamplingOptions<'a, PS, SOP, NewM>>
    where
        PS: ProbabilitySpec,
    {
        if balancing.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            coordination: self.coordination,
            spreading: self.spreading,
            balancing: Some(balancing),
        })
    }

    // BUILDERS
    #[inline]
    pub fn to_probabilities(&self) -> PS::Native
    where
        PS: ProbabilitySpec,
    {
        self.probabilities.to_native(self.eps)
    }
    #[inline]
    pub fn to_probabilities_float(&self) -> FloatProbabilities
    where
        PS: ProbabilitySpec,
    {
        self.probabilities.to_float(self.eps)
    }
    #[inline]
    pub fn to_controller(&self) -> BasicSampleController<PS::Native>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities();
        BasicSampleController::new(probs)
    }
    #[inline]
    pub fn to_controller_float(&self) -> BasicSampleController<FloatProbabilities>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities_float();
        BasicSampleController::new(probs)
    }
    #[inline]
    pub fn to_spreading_controller<N>(
        &self,
    ) -> SamplingOptionsResult<SpreadingSampleController<'_, PS::Native, N, SOP>>
    where
        PS: ProbabilitySpec,
        N: Number,
        SOP: PointSet<N>,
    {
        let controller = self.to_controller();
        let spreading = self.spreading()?;
        SpreadingSampleController::new(controller, spreading)
    }
    #[inline]
    pub fn to_spreading_controller_float<N>(
        &self,
    ) -> SamplingOptionsResult<SpreadingSampleController<'_, FloatProbabilities, N, SOP>>
    where
        PS: ProbabilitySpec,
        N: Number,
        SOP: PointSet<N>,
    {
        let controller = self.to_controller_float();
        let spreading = self.spreading()?;
        SpreadingSampleController::new(controller, spreading)
    }
}

impl<'a> SamplingOptions<'a, ProbabilitySpecUnequal<'a>> {
    #[inline]
    pub fn new(
        probabilities: &'a [f64],
    ) -> SamplingOptionsResult<SamplingOptions<'a, ProbabilitySpecUnequal<'a>, (), ()>> {
        Self::with_spec(ProbabilitySpecUnequal::new(probabilities)?)
    }
    #[inline]
    pub fn with_spec(
        spec: ProbabilitySpecUnequal<'a>,
    ) -> SamplingOptionsResult<SamplingOptions<'a, ProbabilitySpecUnequal<'a>, (), ()>> {
        const EPS: f64 = 1e-9;
        Ok(SamplingOptions {
            probabilities: spec,
            eps: EPS,
            max_iterations: NonZeroUsize::new(1000).unwrap(),
            coordination: None,
            spreading: None,
            balancing: None,
        })
    }
}
impl<'a> SamplingOptions<'a, ProbabilitySpecEqual> {
    #[inline]
    pub fn new_equal(
        population_size: NonZeroUsize,
        sample_size: usize,
    ) -> SamplingOptionsResult<SamplingOptions<'static, ProbabilitySpecEqual, (), ()>> {
        Self::with_spec_equal(ProbabilitySpecEqual::new(population_size, sample_size)?)
    }
    #[inline]
    pub fn with_spec_equal(
        spec: ProbabilitySpecEqual,
    ) -> SamplingOptionsResult<SamplingOptions<'static, ProbabilitySpecEqual, (), ()>> {
        Ok(SamplingOptions {
            probabilities: spec,
            eps: 1e-9,
            max_iterations: NonZeroUsize::new(1000).unwrap(),
            coordination: None,
            spreading: None,
            balancing: None,
        })
    }
}

impl<'a> TryFrom<&'a [f64]> for SamplingOptions<'a, ProbabilitySpecUnequal<'a>, (), ()> {
    type Error = SamplingOptionsError;
    #[inline]
    fn try_from(probabilities: &'a [f64]) -> Result<Self, Self::Error> { Self::new(probabilities) }
}
impl TryFrom<(NonZeroUsize, usize)> for SamplingOptions<'static, ProbabilitySpecEqual, (), ()> {
    type Error = SamplingOptionsError;
    #[inline]
    fn try_from(
        (population_size, sample_size): (NonZeroUsize, usize),
    ) -> Result<Self, Self::Error> {
        Self::new_equal(population_size, sample_size)
    }
}
