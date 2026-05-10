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

use std::borrow::Cow;
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

use crate::kd_tree::Tree;
use crate::matrix::Dimensions;
use crate::probabilities::FloatProbabilities;
use crate::sample_controller::SampleController;
use crate::spatial::PointSet;

#[must_use]
#[derive(Clone, Debug)]
pub struct SamplingOptions<PS, AUX = (), BAL = ()> {
    /// Probability specification
    probabilities: PS,
    /// Epsilon to be used in float comparisons
    eps: f64,
    /// Maximum number of iterations to run for an algorithm
    max_iterations: NonZeroUsize,
    /// Spreading auxiliaries options
    // spreading: Option<SpreadingOptions<SOP>>,
    spreading: AUX,
    /// Balancing auxiliaries options
    balancing: BAL,
}

// ACCESSORS
impl<PS, AUX, BAL> SamplingOptions<PS, AUX, BAL> {
    #[inline]
    pub fn probabilities(&self) -> &PS { &self.probabilities }
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize
    where
        PS: ProbabilitySpec,
    {
        self.probabilities().population_size()
    }
    #[must_use]
    #[inline]
    pub fn sample_size(&self) -> usize
    where
        PS: ProbabilitySpec,
    {
        self.probabilities().sample_size()
    }
    #[must_use]
    #[inline]
    pub fn eps(&self) -> f64 { self.eps }
    #[must_use]
    #[inline]
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
}
impl<PS, AUXP, BAL> SamplingOptions<PS, SpreadingOptions<AUXP>, BAL> {
    #[inline]
    pub fn spreading(&self) -> &SpreadingOptions<AUXP> { &self.spreading }
}
impl<PS, AUX, BALP> SamplingOptions<PS, AUX, BalancingOptions<BALP>> {
    #[inline]
    pub fn balancing(&self) -> &BalancingOptions<BALP> { &self.balancing }
}

// SETTERS
impl<PS, AUX, BAL> SamplingOptions<PS, AUX, BAL> {
    /// Sets the epsilon value, a value to be used for float comparisons.
    ///
    /// # Errors
    /// Returns an error if `eps` is not contained within $[0.0, 0.001)$.
    #[inline]
    pub fn set_eps(mut self, eps: f64) -> SamplingOptionsResult<Self> {
        if !(0.0..0.001).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        self.eps = eps;
        Ok(self)
    }
    /// Sets the maximum number of iterations to be used.
    ///
    /// # Errors
    /// Returns an error if `max` cannot be converted into a [`NonZeroUsize`].
    #[inline]
    pub fn set_max_iterations<NZ>(mut self, max: NZ) -> SamplingOptionsResult<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        self.max_iterations = max
            .try_into()
            .map_err(|_| SamplingOptionsError::InvalidIterations)?;
        Ok(self)
    }
    /// Sets the spreading options
    ///
    /// # Errors
    /// Returns an error if the size of `data` does not match population size
    #[inline]
    pub fn set_spreading<AUXP, I>(
        self,
        data: I,
    ) -> SamplingOptionsResult<SamplingOptions<PS, SpreadingOptions<AUXP>, BAL>>
    where
        PS: ProbabilitySpec,
        AUXP: PointSet,
        I: Into<SpreadingOptions<AUXP>>,
    {
        let data = data.into();
        if data.data().size() != self.population_size() {
            return Err(SamplingOptionsError::InvalidSpreading);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            spreading: data,
            balancing: self.balancing,
        })
    }
    /// Sets the balancing options
    ///
    /// # Errors
    /// Returns an error if `data.size()` does not match population size
    #[inline]
    pub fn set_balancing<BALP, I>(
        self,
        data: I,
    ) -> SamplingOptionsResult<SamplingOptions<PS, AUX, BalancingOptions<BALP>>>
    where
        PS: ProbabilitySpec,
        BALP: Dimensions,
        I: Into<BalancingOptions<BALP>>,
    {
        let data = data.into();
        if data.data().nrow() != self.population_size() {
            return Err(SamplingOptionsError::InvalidBalancing);
        }
        Ok(SamplingOptions {
            probabilities: self.probabilities,
            eps: self.eps,
            max_iterations: self.max_iterations,
            spreading: self.spreading,
            balancing: data,
        })
    }
}

impl<PS, AUX, BAL> SamplingOptions<PS, AUX, BAL> {
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
    pub fn to_controller(&self) -> SampleController<PS::Native, ()>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities();
        SampleController::new(probs)
    }
    #[inline]
    pub fn to_controller_float(&self) -> SampleController<FloatProbabilities, ()>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities_float();
        SampleController::new(probs)
    }
}
impl<PS, AUXP, BAL> SamplingOptions<PS, SpreadingOptions<AUXP>, BAL>
where
    AUXP: PointSet,
{
    #[inline]
    pub fn to_spreading_controller(&self) -> SampleController<PS::Native, Tree<'_, AUXP>>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
    #[inline]
    pub fn to_spreading_controller_float(
        &self,
    ) -> SampleController<FloatProbabilities, Tree<'_, AUXP>>
    where
        PS: ProbabilitySpec,
    {
        let probs = self.to_probabilities_float();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
}

/// Default epsilon value
const EPS: f64 = 1e-9;
/// Default maximum iterations value
const MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(1000).expect("infallible");

impl<'bprob> SamplingOptions<ProbabilitySpecUnequal<'bprob>> {
    /// Initializes `SamplingOptions` by some probability container.
    ///
    /// # Errors
    /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
    #[inline]
    pub fn new(
        probabilities: Cow<'bprob, [f64]>,
    ) -> SamplingOptionsResult<SamplingOptions<ProbabilitySpecUnequal<'bprob>, (), ()>> {
        Ok(Self::with_spec(ProbabilitySpecUnequal::new(probabilities)?))
    }
    #[inline]
    pub fn with_spec(
        spec: ProbabilitySpecUnequal<'bprob>,
    ) -> SamplingOptions<ProbabilitySpecUnequal<'bprob>, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: EPS,
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}
impl SamplingOptions<ProbabilitySpecEqual> {
    /// Initializes `SamplingOptions` by some `population_size`, `sample_size` pair.
    ///
    /// # Errors
    /// If the pair cannot be turned into [`ProbabilitySpecEqual`].
    #[inline]
    pub fn new_equal<NZ>(
        population_size: NZ,
        sample_size: usize,
    ) -> SamplingOptionsResult<SamplingOptions<ProbabilitySpecEqual, (), ()>>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        Ok(Self::with_spec_equal(ProbabilitySpecEqual::new(
            population_size,
            sample_size,
        )?))
    }
    #[inline]
    pub fn with_spec_equal(
        spec: ProbabilitySpecEqual,
    ) -> SamplingOptions<ProbabilitySpecEqual, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: EPS,
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}
