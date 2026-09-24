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

//! Methods for configuring sampling algorithms

mod balancing_opts;
mod error;
mod probability_opts;
mod spreading_opts;

use std::num::NonZeroUsize;

pub use balancing_opts::BalancingOptions;
pub use error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
pub use probability_opts::{
    EqualProbabilities,
    ProbabilitiesSpec,
    UnequalProbabilities,
    UnequalProbabilitiesInt,
    UnequalProbabilitiesReal,
};
pub use spreading_opts::SpreadingOptions;

use crate::matrix::Dimensions;
use crate::probabilities::{
    Probability,
    ProbabilitySet,
};
use crate::random::{
    FloatRng,
    Rand,
};
use crate::utils::{
    ConstructableDataView,
    Epsilon,
    Number,
    NumberFloat,
    NumberInt,
    PointSet,
    SliceView,
};

/// RNG requirements for an RNG to be able to operate on a [`ProbabilitySpec`]
pub trait SamplingOptionsRng<PO>: FloatRng + Rand<PO::Value> + Rand<PO::Real>
where
    PO: ProbabilitiesSpec,
{
}
impl<PO, R> SamplingOptionsRng<PO> for R
where
    PO: ProbabilitiesSpec,
    R: FloatRng + Rand<PO::Value> + Rand<PO::Real>,
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
    /// Returns a reference to the spreading options
    #[inline]
    pub fn spreading(&self) -> &AUX { &self.spreading }
    /// Returns a reference to the balancing options
    #[inline]
    pub fn balancing(&self) -> &BAL { &self.balancing }
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
    /// Returns an error if ids of probabilities does not exist in spreading.
    #[inline]
    pub fn set_spreading<AUXP, I>(
        self,
        data: I,
    ) -> SamplingOptionsResult<SamplingOptions<PO, SpreadingOptions<AUXP>, BAL>>
    where
        AUXP: PointSet<Id = PO::Id>,
        I: Into<SpreadingOptions<AUXP>>,
    {
        let data = data.into();
        if !self
            .probabilities()
            .ids()
            .all(|id| data.data().contains(id))
        {
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
        BALP: PointSet<Id = PO::Id>,
        I: Into<BalancingOptions<BALP>>,
    {
        let data = data.into();
        if !self
            .probabilities()
            .ids()
            .all(|id| data.data().contains(id))
        {
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
    /// Constructs a [`ProbabilitySet`] from the probability specification
    #[inline]
    pub fn to_probabilityset(
        &self,
    ) -> ProbabilitySet<PO::ConstructableContainer<Probability<PO::Value>>, PO::Value>
    where
        PO: ConstructableDataView,
    {
        ProbabilitySet::from_opts(&self.probabilities, self.eps)
    }
    /// Constructs a real-valued [`ProbabilitySet`] from the probability specification
    #[inline]
    pub fn to_probabilityset_real(
        &self,
    ) -> ProbabilitySet<PO::ConstructableContainer<Probability<PO::Real>>, PO::Real>
    where
        PO: ConstructableDataView,
    {
        ProbabilitySet::from_opts_real(&self.probabilities, self.eps)
    }
}

/// Default maximum iterations value
const MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(1000).expect("infallible");

impl<PS> SamplingOptions<PS>
where
    PS: ProbabilitiesSpec,
{
    /// Initializes `SamplingOptions` with unequal probability options
    #[inline]
    pub fn with_spec(spec: PS) -> SamplingOptions<PS, (), ()> {
        SamplingOptions {
            probabilities: spec,
            eps: Epsilon::<PS::Real>::default(),
            max_iterations: MAX_ITERATIONS,
            spreading: (),
            balancing: (),
        }
    }
}
impl<PD> SamplingOptions<UnequalProbabilities<UnequalProbabilitiesReal<PD>>>
where
    PD: SliceView,
    PD::Value: NumberFloat,
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
    PD::Value: NumberInt,
{
    /// Initializes `SamplingOptions` by some probability container.
    ///
    /// # Errors
    /// If `probabilities` cannot be turned into [`ProbabilitySpecUnequal`].
    #[inline]
    pub fn new_int(probabilities: PD, max: PD::Value) -> SamplingOptionsResult<Self> {
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
        let population_size = population_size
            .try_into()
            .map_err(|_| SamplingOptionsError::InvalidPopulationSize)?;
        let spec = EqualProbabilities::new(population_size, sample_size)?;
        Ok(Self::with_spec(spec))
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
}
