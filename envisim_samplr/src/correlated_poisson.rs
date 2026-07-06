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

//! Correlated poisson designs
//!
//! Implements [`CorrelatedPoissonSampling`] and [`SpatialCorrelatedPoissonSampling`] for
//! [`SamplingOptions`].
//!
//! # References
//! Bondesson, L., & Thorburn, D. (2008).
//! A list sequential sampling method suitable for real‐time sampling.
//! Scandinavian Journal of Statistics, 35(3), 466-483.
//! <https://doi.org/10.1111/j.1467-9469.2008.00596.x>
//!
//! Grafström, A. (2012).
//! Spatially correlated Poisson sampling.
//! Journal of Statistical Planning and Inference, 142(1), 139-147.
//! <https://doi.org/10.1016/j.jspi.2011.07.003>
//!
//! Prentius, W. (2024).
//! Locally correlated Poisson sampling.
//! Environmetrics, 35(2), e2832.
//! <https://doi.org/10.1002/env.2832>

use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::WeightedSearcher;
use envisim_utils::kd_tree::searcher::neighbour::WeightedNeighbour;
use envisim_utils::probabilities::Probability;
use envisim_utils::random::{
    FloatRng,
    Rand,
    Rng,
    random_element,
};
use envisim_utils::sample_controller::{
    SampleController,
    UnitRemoving,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    CoordinationOptions,
    ProbabilityOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::{
    Number,
    PointSet,
};
use num_traits::ToPrimitive;

pub use crate::error::SamplingError;
use crate::error::SamplingResult;

pub trait CorrelatedPoissonStrategy<TREE> {
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: Rand<f64>;
    fn select_unit<R>(
        &mut self,
        controller: &mut SampleController<f64, TREE>,
        rng: &mut R,
    ) -> Option<usize>
    where
        R: Rand<usize>;
    fn update_probabilities(
        &mut self,
        controller: &mut SampleController<f64, TREE>,
        id: usize,
        probability: Probability<f64>,
        quota: f64,
    );
}

#[must_use]
pub struct CorrelatedPoissonRunner<S, TREE>
where
    SampleController<f64, TREE>: UnitRemoving,
{
    /// Sample controller
    controller: SampleController<f64, TREE>,
    /// Sample strategy
    strategy: S,
}

impl<S, TREE> CorrelatedPoissonRunner<S, TREE>
where
    S: CorrelatedPoissonStrategy<TREE>,
    SampleController<f64, TREE>: UnitRemoving,
{
    /// Runs the simulation and returns a sorted sample
    #[must_use]
    #[inline]
    fn sample<R>(mut self, rng: &mut R) -> Vec<usize>
    where
        R: Rand<f64> + Rand<usize>,
    {
        self.run(rng);
        self.controller.to_sorted_sample_vec()
    }
    /// Runs the sampling algorithm
    #[inline]
    fn run<R>(&mut self, rng: &mut R)
    where
        R: Rand<f64> + Rand<usize>,
    {
        while let Some(id) = self.strategy.select_unit(&mut self.controller, rng) {
            let (p, q) = self.decide_unit(rng, id);
            self.strategy
                .update_probabilities(&mut self.controller, id, p, q);
        }
    }
    /// Decides the outcome of the selected unit
    #[inline]
    fn decide_unit<R>(&mut self, rng: &mut R, id: usize) -> (Probability<f64>, f64)
    where
        R: Rand<f64>,
    {
        let probability = self.controller.probabilities()[id];
        let mut quota = probability.get();
        let rv = self.strategy.random_value(rng, id);

        if rv < probability.get() {
            self.controller.unit_set_full(id);
            quota -= 1.0;
        } else {
            self.controller.unit_set_zero(id);
        }

        (probability, quota)
    }
}

#[must_use]
pub struct SequentialStrategy<'bcoord> {
    /// Random values to be used
    random_values: CoordinationOptions<'bcoord>,
    /// Selected unit, used to control the decision order if random values were provided
    unit: usize,
}
impl<'bcoord> SequentialStrategy<'bcoord> {
    /// Constructs a new CPS runner using the sequential strategy
    #[inline]
    pub fn new<PO, AUX, BAL>(
        options: &SamplingOptions<PO, AUX, BAL>,
    ) -> CorrelatedPoissonRunner<Self, ()>
    where
        PO: ProbabilityOptions<Real = f64>,
    {
        let controller = options.to_controller_real();
        CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values: CoordinationOptions::new_empty(),
                unit: 0,
            },
        }
    }
    /// Constructs a new coordinated CPS runner using the sequential strategy
    ///
    /// # Errors
    /// Returns an error if fewer than `population_size` random values is provided.
    #[inline]
    pub fn new_coord<PO, AUX, BAL, C>(
        options: &SamplingOptions<PO, AUX, BAL>,
        random_values: C,
    ) -> SamplingResult<CorrelatedPoissonRunner<Self, ()>>
    where
        PO: ProbabilityOptions<Real = f64>,
        C: Into<CoordinationOptions<'bcoord>>,
    {
        let controller = options.to_controller_real();
        let random_values = random_values.into();
        random_values.check(controller.population_size())?;
        Ok(CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values,
                unit: 0,
            },
        })
    }
}
impl CorrelatedPoissonStrategy<()> for SequentialStrategy<'_> {
    #[inline]
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: Rand<f64>,
    {
        self.random_values.get_or(id, rng)
    }
    #[inline]
    fn select_unit<R>(
        &mut self,
        controller: &mut SampleController<f64, ()>,
        _rng: &mut R,
    ) -> Option<usize>
    where
        R: Rand<usize>,
    {
        // Tempting to use controller.indices().last(), but order is not guaranteed as swap_remove
        // might move a unit forward in the indices.list().

        let pop_size = controller.population_size().get();
        while self.unit < pop_size && !controller.indices().contains(self.unit) {
            self.unit += 1;
        }

        (self.unit < pop_size).then_some(self.unit)
    }
    #[inline]
    fn update_probabilities(
        &mut self,
        controller: &mut SampleController<f64, ()>,
        _id: usize,
        probability: Probability<f64>,
        quota: f64,
    ) {
        if controller.indices().is_empty() {
            return;
        }

        let pop_size = controller.population_size();
        let mut remaining_weight: f64 = 1.0;

        for id_n in (self.unit + 1)..pop_size.get() {
            if !controller.indices().contains(id_n) {
                continue;
            }
            let possible_weight = controller.probabilities().weight_to(probability, id_n);
            let weight = possible_weight.min(remaining_weight);
            let delta = weight * quota;
            let _prest = controller.unit_add_delta_and_decide(id_n, delta);
            remaining_weight -= possible_weight;
            if remaining_weight <= 0.0 {
                break;
            }
        }
    }
}

pub struct SpatialStrategy<'bcoord, P>
where
    P: PointSet,
{
    /// Sample controller
    random_values: CoordinationOptions<'bcoord>,
    /// Order is used together with `random_values`, in order to ensure that the selection order is
    /// the same. If no random values (no coordination), the order is random.
    order: usize,
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: WeightedSearcher<P>,
}
impl<'bcoord, P> SpatialStrategy<'bcoord, P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new CPS runner using the spatial strategy
    #[inline]
    pub fn new<PO, BAL>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    ) -> CorrelatedPoissonRunner<Self, Tree<'_, P>>
    where
        PO: ProbabilityOptions<Real = f64>,
    {
        let controller = options.to_spreading_controller_real();
        let searcher = WeightedSearcher::new(controller.tree().data());
        CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values: CoordinationOptions::new_empty(),
                order: 0,
                searcher,
            },
        }
    }
    /// Constructs a new coordinated CPS runner using the spatial strategy
    ///
    /// # Errors
    /// Returns an error if fewer than `population_size` random values is provided.
    #[inline]
    pub fn new_coord<PO, BAL, C>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
        random_values: C,
    ) -> SamplingResult<CorrelatedPoissonRunner<Self, Tree<'_, P>>>
    where
        PO: ProbabilityOptions<Real = f64>,
        C: Into<CoordinationOptions<'bcoord>>,
    {
        let controller = options.to_spreading_controller_real();
        let searcher = WeightedSearcher::new(controller.tree().data());
        let random_values = random_values.into();
        random_values.check(controller.population_size())?;
        Ok(CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values,
                order: 0,
                searcher,
            },
        })
    }
}
/// Fins the neighbours of the selected unit, and updates their probabilities and selection status
#[inline]
fn spatial_update_probabilities<P>(
    searcher: &mut WeightedSearcher<P>,
    controller: &mut SampleController<f64, Tree<'_, P>>,
    id: usize,
    probability: Probability<f64>,
    quota: f64,
) where
    P: PointSet<Id = usize>,
{
    if controller.indices().is_empty() {
        return;
    }

    searcher
        .reset_from_unit(controller.tree().data(), id, probability.get())
        .expect("id to exist")
        .search(controller.tree(), controller.probabilities())
        .expect("nn to be found");

    let mut remaining_weight: f64 = 1.0;

    let max_distance = searcher
        .max_distance()
        .expect("searcher to have found a neighbour");
    // Units with distance below max_distance are guaranteed their weight
    let guaranteed_units = searcher
        .neighbours()
        .partition_point(|n| n.distance() < max_distance);

    for n in &searcher.neighbours()[0..guaranteed_units] {
        let delta = n.weight() * quota;
        let _prest = controller.unit_add_delta_and_decide(n.id(), delta);
        remaining_weight -= n.weight();
    }

    let sum_of_tie_weights: f64 = searcher.neighbours()[guaranteed_units..]
        .iter()
        .map(WeightedNeighbour::weight)
        .sum();

    #[expect(clippy::float_cmp, reason = "shortcut in case of the unlikely")]
    if sum_of_tie_weights == remaining_weight {
        // Add everything left, if it's exactly solved (unlikely)
        for n in &searcher.neighbours()[guaranteed_units..] {
            let delta = n.weight() * quota;
            let _prest = controller.unit_add_delta_and_decide(n.id(), delta);
            // remaining_weight -= n.weight();
        }
        return;
    }

    // Since the weights are sorted by distance, then weights, we'll try to remove weights by
    // removing as much as possible of the remaining shares, going from the smallest weight to
    // the largest.
    // It might be tempting to add a case before for when only one unit remains, but the only
    // thing we could save on below is a division by 1.0, as we don't know how much weight can
    // be used.
    #[expect(clippy::unwrap_used, reason = "usize to f64 conversion")]
    let mut number_of_shares = (searcher.neighbours().len() - guaranteed_units)
        .to_f64()
        .unwrap();
    for n in &searcher.neighbours()[guaranteed_units..] {
        let removable_weight = n.weight().min(remaining_weight / number_of_shares);
        let delta = remaining_weight * quota;
        let _prest = controller.unit_add_delta_and_decide(n.id(), delta);
        remaining_weight -= removable_weight;
        number_of_shares -= 1.0;
    }
}

impl<P> CorrelatedPoissonStrategy<Tree<'_, P>> for SpatialStrategy<'_, P>
where
    P: PointSet<Id = usize>,
{
    #[must_use]
    #[inline]
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: Rand<f64>,
    {
        self.random_values.get_or(id, rng)
    }
    #[must_use]
    #[inline]
    fn select_unit<R>(
        &mut self,
        controller: &mut SampleController<f64, Tree<'_, P>>,
        rng: &mut R,
    ) -> Option<usize>
    where
        R: Rand<usize>,
    {
        if controller.indices().is_empty() {
            return None;
        }

        if self.random_values.is_empty() {
            return controller.indices().draw(rng);
        }

        // If random values are used -- i.e. coordination -- we want to have selection order fixed,
        // in order to reduce entropy
        let pop_size = controller.population_size().get();
        while self.order < pop_size && !controller.indices().contains(self.order) {
            self.order += 1;
        }

        (self.order < pop_size).then_some(self.order)
    }
    #[inline]
    fn update_probabilities(
        &mut self,
        controller: &mut SampleController<f64, Tree<'_, P>>,
        id: usize,
        probability: Probability<f64>,
        quota: f64,
    ) {
        spatial_update_probabilities(&mut self.searcher, controller, id, probability, quota);
    }
}

#[must_use]
pub struct LocalStrategy<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: WeightedSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<usize>,
}
impl<P> LocalStrategy<P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new CPS runner using the local strategy
    #[inline]
    pub fn new<PO, BAL>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    ) -> CorrelatedPoissonRunner<Self, Tree<'_, P>>
    where
        PO: ProbabilityOptions<Real = f64>,
    {
        let controller = options.to_spreading_controller_real();
        let searcher = WeightedSearcher::new(controller.tree().data());
        CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                searcher,
                candidates: Vec::<usize>::with_capacity(20),
            },
        }
    }
}
impl<P> CorrelatedPoissonStrategy<Tree<'_, P>> for LocalStrategy<P>
where
    P: PointSet<Id = usize>,
{
    #[must_use]
    #[inline]
    fn random_value<R>(&mut self, rng: &mut R, _id: usize) -> f64
    where
        R: Rand<f64>,
    {
        rng.rand()
    }
    #[must_use]
    #[inline]
    fn select_unit<R>(
        &mut self,
        controller: &mut SampleController<f64, Tree<'_, P>>,
        rng: &mut R,
    ) -> Option<usize>
    where
        R: Rand<usize>,
    {
        if controller.indices().len() <= 1 {
            return controller.indices().first();
        } else if controller.indices().len() == 2 {
            return controller.indices().draw(rng);
        }

        let mut minimum_distance = P::Value::max_value();
        self.candidates.clear();

        // Loop through all remaining units
        let mut i = 0;
        while i < controller.indices().len() {
            let id = controller.indices()[i];
            self.searcher
                .reset_from_unit(
                    controller.tree().data(),
                    id,
                    controller.probabilities()[id].get(),
                )
                .expect("id to exist")
                .search(controller.tree(), controller.probabilities())
                .expect("nn to be found");
            // We are guaranteed to have at least one neighbour by the if's in the beginning
            let distance = self
                .searcher
                .max_distance()
                .expect("searcher to have found a neighbour");

            if distance < minimum_distance {
                self.candidates.clear();
                self.candidates.push(id);
                minimum_distance = distance;
            } else if distance == minimum_distance && distance.is_finite() {
                self.candidates.push(id);
            }

            i += 1;
        }

        random_element(rng, &self.candidates).copied()
    }
    #[inline]
    fn update_probabilities(
        &mut self,
        controller: &mut SampleController<f64, Tree<'_, P>>,
        id: usize,
        probability: Probability<f64>,
        quota: f64,
    ) {
        spatial_update_probabilities(&mut self.searcher, controller, id, probability, quota);
    }
}

pub trait CorrelatedPoissonSampling<R>
where
    R: Rng,
{
    /// Draw a sample using the (sequential) correlated poisson sampling method.
    /// A variant of the cps where unit competes in order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.cps(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn cps(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using the (sequential) correlated poisson sampling method.
    /// A variant of the cps where unit competes in order.
    ///
    /// `random_values` are used in order to decide the inclusions of units, allowing for coordination
    /// between multiple sampling efforts.
    ///
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let rv: Vec<f64> = vec![0.2; 10];
    /// let s = SamplingOptions::new(p)?.cps_coord(&mut rng, rv)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if fewer than `population_size` random values is provided.
    /// # Errors
    /// Returns an error if fewer than `population_size` random values is provided.
    fn cps_coord<'bcoord, C>(&self, rng: &mut R, random_values: C) -> SamplingResult<Vec<usize>>
    where
        C: Into<CoordinationOptions<'bcoord>>;
}
pub trait SpatiallyCorrelatedPoissonSampling<R>
where
    R: Rng,
{
    /// Draw a sample using the spatially correlated poisson sampling method.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p)?.set_spreading(m)?.scps(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn scps(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using the spatially correlated poisson sampling method.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// `random_values` are used in order to decide the inclusions of units, allowing for coordination
    /// between multiple sampling efforts.
    ///
    /// ```
    /// # use envisim_samplr::correlated_poisson::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let rv: Vec<f64> = vec![0.2; 10];
    /// let s = SamplingOptions::new(p)?
    ///     .set_spreading(m)?
    ///     .scps_coord(&mut rng, rv)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if fewer than `population_size` random values is provided.
    fn scps_coord<'bcoord, C>(&self, rng: &mut R, random_values: C) -> SamplingResult<Vec<usize>>
    where
        C: Into<CoordinationOptions<'bcoord>>;
    /// Draw a sample using the locally correlated poisson sampling method.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p)?.set_spreading(m)?.lcps(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn lcps(&self, rng: &mut R) -> Vec<usize>;
}
impl<R, PO, AUX, BAL> CorrelatedPoissonSampling<R> for SamplingOptions<PO, AUX, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions<Real = f64>,
{
    #[inline]
    fn cps(&self, rng: &mut R) -> Vec<usize> { SequentialStrategy::new(self).sample(rng) }
    #[inline]
    fn cps_coord<'bcoord, C>(&self, rng: &mut R, random_values: C) -> SamplingResult<Vec<usize>>
    where
        R: FloatRng,
        C: Into<CoordinationOptions<'bcoord>>,
    {
        Ok(SequentialStrategy::new_coord(self, random_values)?.sample(rng))
    }
}
impl<R, PO, P, BAL> SpatiallyCorrelatedPoissonSampling<R>
    for SamplingOptions<PO, SpreadingOptions<P>, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize>,
{
    #[inline]
    fn scps(&self, rng: &mut R) -> Vec<usize> { SpatialStrategy::new(self).sample(rng) }
    #[inline]
    fn scps_coord<'bcoord, C>(&self, rng: &mut R, random_values: C) -> SamplingResult<Vec<usize>>
    where
        C: Into<CoordinationOptions<'bcoord>>,
    {
        Ok(SpatialStrategy::new_coord(self, random_values)?.sample(rng))
    }
    #[inline]
    fn lcps(&self, rng: &mut R) -> Vec<usize> { LocalStrategy::new(self).sample(rng) }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use envisim_utils::random::*;
    use envisim_utils::sampling_options::CoordinationOptions;
    use envisim_utils::test_utils::*;

    use super::*;

    const RV_0: [f64; 10] = [0.0; 10];
    const RV_1: [f64; 10] = [1.0; 10];

    fn coord_0() -> CoordinationOptions<'static> { Cow::from(&RV_0).into() }
    fn coord_1() -> CoordinationOptions<'static> { Cow::from(&RV_1).into() }

    #[test]
    fn cps_sampler() -> SamplingResult<()> {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = Data10::options_e();
        let mut cps = SequentialStrategy::new_coord(&options, coord_0()).unwrap();
        assert_eq!(
            cps.decide_unit(&mut rng, 7),
            (Probability::Partial(0.2), -0.8)
        );

        let mut cps = SequentialStrategy::new_coord(&options, coord_1()).unwrap();
        assert_eq!(
            cps.decide_unit(&mut rng, 7),
            (Probability::Partial(0.2), 0.2)
        );
        Ok(())
    }

    fn decide_and_update<'a, R, S, TREE>(
        cps: &mut CorrelatedPoissonRunner<S, TREE>,
        rng: &mut R,
        id: usize,
    ) -> (Probability<f64>, f64)
    where
        R: FloatRng,
        S: CorrelatedPoissonStrategy<TREE>,
        SampleController<f64, TREE>: UnitRemoving,
    {
        let (p, q) = cps.decide_unit(rng, id);
        cps.strategy
            .update_probabilities(&mut cps.controller, id, p, q);
        (p, q)
    }

    #[test]
    fn cps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = Data10::options_e();
        let mut cpsv = SequentialStrategy::new_coord(&options, coord_0()).unwrap();
        let _ = decide_and_update(&mut cpsv, &mut rng, 0);
        assert_vec!(
            cpsv.controller.probabilities().to_raw()[1..=4],
            vec![0.0; 4]
        );

        let options = Data10::options_e();
        let mut cpsv = SequentialStrategy::new_coord(&options, coord_1()).unwrap();
        let _ = decide_and_update(&mut cpsv, &mut rng, 0);
        assert_vec!(
            cpsv.controller.probabilities().to_raw()[1..=4],
            vec![0.25; 4]
        );

        // let options = options_ue();
        println!("CPS1");
        let mut rng = SmallRng::seed_from_u64(42);
        let options = Data10::options_e();
        let s = options.cps(&mut rng);
        assert_eq!(s.len(), 2);
        println!("CPS2");
        let options = Data10::options_e();
        let s = options.cps(&mut rng);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn scps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = Data10::options_e();
        let mut cps = SpatialStrategy::new_coord(&options, coord_0()).unwrap();
        let _ = decide_and_update(&mut cps, &mut rng, 0);
        println!("{:?}", cps.controller.probabilities().data());
        assert_delta!(cps.controller.probabilities()[1].get(), 0.0);
        assert_delta!(cps.controller.probabilities()[8].get(), 0.0);
        assert_delta!(cps.controller.probabilities()[4].get(), 0.0);
        assert_delta!(cps.controller.probabilities()[2].get(), 0.0);

        let mut cps = SpatialStrategy::new_coord(&options, coord_1()).unwrap();
        let _ = decide_and_update(&mut cps, &mut rng, 9);
        assert_delta!(cps.controller.probabilities()[4].get(), 0.25);
        assert_delta!(cps.controller.probabilities()[2].get(), 0.25);
        assert_delta!(cps.controller.probabilities()[0].get(), 0.25);
        assert_delta!(cps.controller.probabilities()[7].get(), 0.25);
    }

    #[test]
    fn lcps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let options = Data10::options_e();
        let mut cps = LocalStrategy::new(&options);
        assert_eq!(
            cps.strategy.select_unit(&mut cps.controller, &mut rng),
            Some(8)
        );
    }
}
