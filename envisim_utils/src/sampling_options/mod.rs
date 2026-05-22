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
mod epsilon;
mod error;
mod probability_opts;
mod spreading_opts;

use std::borrow::Cow;
use std::num::NonZeroUsize;

pub use balancing_opts::BalancingOptions;
pub use coordination_opts::CoordinationOptions;
pub use epsilon::Epsilon;
pub use error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use probability_opts::{
    EqualProbabilityOptions,
    ProbabilityOptions,
    UnequalProbabilityOptions,
};
pub use spreading_opts::SpreadingOptions;

use crate::kd_tree::Tree;
use crate::matrix::Dimensions;
use crate::probabilities::ProbabilitySet;
use crate::sample_controller::SampleController;
use crate::spatial::PointSet;

#[must_use]
#[derive(Clone, Debug)]
pub struct SamplingOptions<PO, AUX = (), BAL = ()>
where
    PO: ProbabilityOptions,
{
    /// Probability specification
    probabilities: PO,
    /// Epsilon to be used in float comparisons
    eps: Epsilon<PO::Real>,
    /// Maximum number of iterations to run for an algorithm
    max_iterations: NonZeroUsize,
    /// Spreading auxiliaries options
    // spreading: Option<SpreadingOptions<SOP>>,
    spreading: AUX,
    /// Balancing auxiliaries options
    balancing: BAL,
}

// ACCESSORS
impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilityOptions,
{
    #[inline]
    pub fn probabilities(&self) -> &PO { &self.probabilities }
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize { self.probabilities().population_size() }
    #[must_use]
    #[inline]
    pub fn sample_size(&self) -> usize { self.probabilities().sample_size() }
    #[must_use]
    #[inline]
    pub fn eps(&self) -> Epsilon<PO::Real> { self.eps }
    #[must_use]
    #[inline]
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
}
impl<PO, AUXP, BAL> SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>
where
    PO: ProbabilityOptions,
{
    #[inline]
    pub fn spreading(&self) -> &SpreadingOptions<AUXP> { &self.spreading }
}
impl<PO, AUX, BALP> SamplingOptions<PO, AUX, BalancingOptions<BALP>>
where
    PO: ProbabilityOptions,
{
    #[inline]
    pub fn balancing(&self) -> &BalancingOptions<BALP> { &self.balancing }
}

// SETTERS
impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilityOptions,
{
    /// Sets the epsilon value, a value to be used for float comparisons.
    ///
    /// # Errors
    /// Returns an error if `eps` is not contained within $[0.0, 0.001)$.
    #[inline]
    pub fn set_eps<E>(mut self, eps: E) -> SamplingOptionsResult<Self>
    where
        E: TryInto<Epsilon<PO::Real>>,
    {
        self.eps = eps.try_into()?;
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
    ) -> SamplingOptionsResult<SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>>
    where
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
    ) -> SamplingOptionsResult<SamplingOptions<PO, AUX, BalancingOptions<BALP>>>
    where
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

impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilityOptions,
{
    // BUILDERS
    #[inline]
    pub fn to_probabilityset(&self) -> ProbabilitySet<PO::Native> {
        self.probabilities.to_probabilityset(self.eps)
    }
    #[inline]
    pub fn to_probabilityset_real(&self) -> ProbabilitySet<PO::Real> {
        self.probabilities.to_probabilityset_real(self.eps)
    }
    #[inline]
    pub fn to_controller(&self) -> SampleController<PO::Native, ()> {
        let probs = self.to_probabilityset();
        SampleController::new(probs)
    }
    #[inline]
    pub fn to_realcontroller(&self) -> SampleController<PO::Real, ()> {
        let probs = self.to_probabilityset_real();
        SampleController::new(probs)
    }
}
impl<PO, AUXP, BAL> SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>
where
    PO: ProbabilityOptions,
    AUXP: PointSet,
{
    #[inline]
    pub fn to_spreading_controller(&self) -> SampleController<PO::Native, Tree<'_, AUXP>> {
        let probs = self.to_probabilityset();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
    #[inline]
    pub fn to_spreading_controller_real(&self) -> SampleController<PO::Real, Tree<'_, AUXP>> {
        let probs = self.to_probabilityset_real();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
}

/// Default maximum iterations value
const MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(1000).expect("infallible");

impl SamplingOptions<EqualProbabilityOptions> {
    /// Initializes `SamplingOptions` by some `population_size`, `sample_size` pair.
    ///
    /// # Errors
    /// If the pair cannot be turned into [`ProbabilitySpecEqual`].
    #[inline]
    pub fn new_equal<NZ>(
        population_size: NZ,
        sample_size: usize,
    ) -> SamplingOptionsResult<SamplingOptions<EqualProbabilityOptions, (), ()>>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let spec = EqualProbabilityOptions::new(population_size, sample_size)?;
        Ok(Self::with_spec_equal(spec))
    }
    #[inline]
    pub fn with_spec_equal(
        spec: EqualProbabilityOptions,
    ) -> SamplingOptions<EqualProbabilityOptions, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<f64>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}

macro_rules! opts_impl_float {
    ($t:ty) => {
        impl<'bprob> SamplingOptions<UnequalProbabilityOptions<'bprob, $t>> {
            /// Initializes `SamplingOptions` by some probability container.
            ///
            /// # Errors
            /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
            #[inline]
            pub fn new(
                probabilities: Cow<'bprob, [$t]>,
            ) -> SamplingOptionsResult<SamplingOptions<UnequalProbabilityOptions<'bprob, $t>, (), ()>>
            {
                let spec = UnequalProbabilityOptions::new(
                    probabilities,
                )?;
                Ok(Self::with_spec(spec))
            }
            #[inline]
            pub fn with_spec(
                spec: UnequalProbabilityOptions<'bprob, $t>,
            ) -> SamplingOptions<UnequalProbabilityOptions<'bprob, $t>, (), ()> {
                SamplingOptions {
                    probabilities: spec,
                    eps: Epsilon::<$t>::default(),
                    max_iterations: MAX_ITERATIONS,
                    spreading: (),
                    balancing: (),
                }
            }
        }
    };
}

macro_rules! opts_impl_int {
    ($t:ty) => {
        impl<'bprob> SamplingOptions<UnequalProbabilityOptions<'bprob, $t>> {
            /// Initializes `SamplingOptions` by some probability container.
            ///
            /// # Errors
            /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
            #[inline]
            pub fn new(
                probabilities: Cow<'bprob, [$t]>, max: $t
            ) -> SamplingOptionsResult<SamplingOptions<UnequalProbabilityOptions<'bprob, $t>, (), ()>>
            {
                let spec = UnequalProbabilityOptions::new(
                    probabilities,max
                )?;
                Ok(Self::with_spec(spec))
            }
            #[inline]
            pub fn with_spec(
                spec: UnequalProbabilityOptions<'bprob, $t>,
            ) -> SamplingOptions<UnequalProbabilityOptions<'bprob, $t>, (), ()> {
                SamplingOptions {
                    probabilities: spec,
                    eps: Epsilon::<f64>::default(),
                    max_iterations: MAX_ITERATIONS,
                    spreading: (),
                    balancing: (),
                }
            }
        }
    };
}

opts_impl_int!(usize);
opts_impl_int!(u8);
opts_impl_int!(u16);
opts_impl_int!(u32);
opts_impl_int!(u64);
opts_impl_int!(u128);

opts_impl_float!(f32);
opts_impl_float!(f64);
