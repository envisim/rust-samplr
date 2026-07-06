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
mod probability_opts;
mod spreading_opts;

use std::num::NonZeroUsize;

pub use balancing_opts::BalancingOptions;
pub use coordination_opts::CoordinationOptions;
pub use error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
pub use probability_opts::{
    EqualProbabilities,
    ProbabilitiesSpec,
    ProbabilitiesView,
    UnequalProbabilities,
    UnequalProbabilitiesInt,
    UnequalProbabilitiesReal,
};
pub use spreading_opts::SpreadingOptions;

use crate::kd_tree::Tree;
use crate::matrix::Dimensions;
use crate::probabilities::ProbabilitySet;
use crate::random::{
    FloatRng,
    Rand,
};
use crate::sample_controller::SampleController;
use crate::utils::{
    Epsilon,
    Number,
    NumberFloat,
    NumberInt,
    PointSet,
    SliceView,
};

pub trait SamplingOptionsRng<PO>: FloatRng + Rand<PO::Native> + Rand<PO::Real>
where
    PO: ProbabilitiesSpec,
{
}
impl<PO, R> SamplingOptionsRng<PO> for R
where
    PO: ProbabilitiesSpec,
    R: FloatRng + Rand<PO::Native> + Rand<PO::Real>,
{
}

/// Contains the sampling options needed to perform equal/unequal probability sampling using
/// classic, spatially balanced and balanced sampling methods.
#[must_use]
#[derive(Clone, Debug)]
pub struct SamplingOptions<PO, AUX = (), BAL = ()>
where
    PO: ProbabilitiesSpec,
{
    /// Probability specification
    probabilities: PO,
    /// Epsilon to be used in float comparisons
    eps: Epsilon<PO::Real>,
    /// Maximum number of iterations to run for an algorithm
    max_iterations: NonZeroUsize,
    /// Spreading auxiliaries options
    spreading: AUX,
    /// Balancing auxiliaries options
    balancing: BAL,
}

// ACCESSORS
impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilitiesSpec,
{
    /// Returns a reference to the provided probability specification
    #[inline]
    pub fn probabilities(&self) -> &PO { &self.probabilities }
    /// Returns the population size as determined by the probability specification
    #[must_use]
    #[inline]
    pub fn population_size(&self) -> NonZeroUsize { self.probabilities().population_size() }
    /// Returns the sample size as determined by the probability specification
    #[must_use]
    #[inline]
    pub fn sample_size(&self) -> usize { self.probabilities().sample_size() }
    /// Returns the real-valued epsilon
    #[inline]
    pub fn eps(&self) -> Epsilon<PO::Real> { self.eps }
    /// Returns the maximum number of iterations
    #[must_use]
    #[inline]
    pub fn max_iterations(&self) -> NonZeroUsize { self.max_iterations }
}
impl<PO, AUXP, BAL> SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>
where
    PO: ProbabilitiesSpec,
{
    #[inline]
    pub fn spreading(&self) -> &SpreadingOptions<AUXP> { &self.spreading }
}
impl<PO, AUX, BALP> SamplingOptions<PO, AUX, BalancingOptions<BALP>>
where
    PO: ProbabilitiesSpec,
{
    #[inline]
    pub fn balancing(&self) -> &BalancingOptions<BALP> { &self.balancing }
}

// SETTERS
impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilitiesSpec,
{
    /// Sets the epsilon value, a value to be used for float comparisons.
    ///
    /// # Errors
    /// Returns an error if `eps` is not contained within $[0.0, 0.001)$.
    #[inline]
    pub fn set_eps<E>(mut self, eps: E) -> SamplingOptionsResult<Self>
    where
        E: TryInto<Epsilon<PO::Real>, Error = SamplingOptionsError>,
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
        if data.data().len() != self.population_size() {
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

// BUILDERS
impl<PO, AUX, BAL> SamplingOptions<PO, AUX, BAL>
where
    PO: ProbabilitiesSpec,
{
    /// Constructs a [`ProbabilitySet`] from the probability specification
    #[inline]
    pub fn to_probabilityset(&self) -> ProbabilitySet<PO::Native> {
        self.probabilities.to_probabilityset(self.eps)
    }
    /// Constructs a real-valued [`ProbabilitySet`] from the probability specification
    #[inline]
    pub fn to_probabilityset_real(&self) -> ProbabilitySet<PO::Real> {
        self.probabilities.to_probabilityset_real(self.eps)
    }
    /// Constructs a [`SampleController`] from the probability specification
    #[inline]
    pub fn to_controller(&self) -> SampleController<PO::Native, ()> {
        let probs = self.to_probabilityset();
        SampleController::new(probs)
    }
    /// Constructs a real-valued [`SampleController`] from the probability specification
    #[inline]
    pub fn to_controller_real(&self) -> SampleController<PO::Real, ()> {
        let probs = self.to_probabilityset_real();
        SampleController::new(probs)
    }
}
impl<PO, AUXP, BAL> SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>
where
    PO: ProbabilitiesSpec,
    AUXP: PointSet<Id = usize>,
{
    /// Constructs a [`SampleController`] containing a KD-tree from the probability specification.
    #[inline]
    pub fn to_spreading_controller(&self) -> SampleController<PO::Native, Tree<'_, AUXP>> {
        let probs = self.to_probabilityset();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
    /// Constructs a real-valued [`SampleController`] containing a KD-tree from the probability
    /// specification.
    #[inline]
    pub fn to_spreading_controller_real(&self) -> SampleController<PO::Real, Tree<'_, AUXP>> {
        let probs = self.to_probabilityset_real();
        let spreading = self.spreading();
        SampleController::new_spreading(probs, spreading)
    }
}

/// Default maximum iterations value
const MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(1000).expect("infallible");

impl<UPO> SamplingOptions<UnequalProbabilities<UPO>>
where
    UPO: ProbabilitiesSpec,
{
    /// Initializes `SamplingOptions` with unequal probability options
    #[inline]
    pub fn with_spec(
        spec: UnequalProbabilities<UPO>,
    ) -> SamplingOptions<UnequalProbabilities<UPO>, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<UPO::Real>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}
impl<PD> SamplingOptions<UnequalProbabilities<UnequalProbabilitiesReal<PD>>>
where
    PD: SliceView,
    PD::Elem: NumberFloat,
{
    /// Initializes `SamplingOptions` by some probability container.
    ///
    /// # Errors
    /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
    #[inline]
    pub fn new(probabilities: PD) -> SamplingOptionsResult<Self> {
        let spec = UnequalProbabilities::new(probabilities)?;
        Ok(Self::with_spec(spec))
    }
}
impl<PD> SamplingOptions<UnequalProbabilities<UnequalProbabilitiesInt<PD>>>
where
    PD: SliceView,
    PD::Elem: NumberInt,
{
    /// Initializes `SamplingOptions` by some probability container.
    ///
    /// # Errors
    /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
    #[inline]
    pub fn new_int(probabilities: PD, max: PD::Elem) -> SamplingOptionsResult<Self> {
        if !max.is_pos_finite() {
            return Err(SamplingOptionsError::InvalidProbability);
        }
        let spec = UnequalProbabilities::new_int(probabilities, max)?;
        Ok(Self::with_spec(spec))
    }
}

impl SamplingOptions<EqualProbabilities> {
    /// Initializes `SamplingOptions` by some `population_size`, `sample_size` pair.
    ///
    /// # Errors
    /// If the pair cannot be turned into [`ProbabilitySpecEqual`].
    #[inline]
    pub fn new_equal<NZ>(population_size: NZ, sample_size: usize) -> SamplingOptionsResult<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let spec = EqualProbabilities::new(population_size, sample_size)?;
        Ok(Self::with_spec_equal(spec))
    }
    /// Initializes `SamplingOptions` with spreading `data` and an equal probability specification
    /// determined by the size of the spreading `data` and the `sample_size`.
    ///
    /// # Errors
    /// If [`ProbabilitySpecEqual`] cannot be constructed, i.e. if `sample_size` is larger than the
    /// population size.
    #[inline]
    pub fn with_spreading<AUXP, I>(
        data: I,
        sample_size: usize,
    ) -> SamplingOptionsResult<SamplingOptions<EqualProbabilities, SpreadingOptions<AUXP>, ()>>
    where
        AUXP: PointSet,
        I: Into<SpreadingOptions<AUXP>>,
    {
        let data = data.into();
        let population_size = data.data().len();
        let spec = EqualProbabilities::new(population_size, sample_size)?;
        Ok(SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<f64>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: data,
            balancing: (),
        })
    }
    /// Initializes `SamplingOptions` with balancing `data` and an equal probability specification
    /// determined by the size of the balancing `data` and the `sample_size`.
    ///
    /// # Errors
    /// If [`ProbabilitySpecEqual`] cannot be constructed, i.e. if `sample_size` is larger than the
    /// population size.
    #[inline]
    pub fn with_balancing<BALP, I>(
        data: I,
        sample_size: usize,
    ) -> SamplingOptionsResult<SamplingOptions<EqualProbabilities, (), BalancingOptions<BALP>>>
    where
        BALP: Dimensions,
        I: Into<BalancingOptions<BALP>>,
    {
        let data = data.into();
        let population_size = data.data().nrow();
        let spec = EqualProbabilities::new(population_size, sample_size)?;
        Ok(SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<f64>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: data,
        })
    }
    /// Initializes `SamplingOptions` with equal probability options
    #[inline]
    pub fn with_spec_equal(
        spec: EqualProbabilities,
    ) -> SamplingOptions<EqualProbabilities, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<f64>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}
