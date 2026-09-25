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
use envisim_utils::kd_tree::searcher::{
    NeighbourView,
    WeightCollection,
    WeightedSearcher,
};
use envisim_utils::probabilities::{
    Probability,
    ProbabilityStore,
};
use envisim_utils::random::{
    Rand,
    Rng,
    random_element,
};
use envisim_utils::sample_controller::{
    SampleController,
    TreeStorage,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    ProbabilitiesSpec,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::{
    DataView,
    Number,
    PointSet,
};
use num_traits::{
    ConstOne,
    ToPrimitive,
};

pub use crate::SamplingError;
use crate::SamplingResult;

/// Finds the first unit after `curr` that can be selected
fn select_unit_sequential<PR, TR>(
    curr: Option<PR::Id>,
    controller: &SampleController<PR, TR>,
) -> Option<PR::Id>
where
    PR: ProbabilityStore,
{
    // Empty means empty
    if controller.indices().is_empty() {
        return None;
    }

    // For None curr, find first match
    let Some(unit) = curr else {
        return controller
            .probabilities()
            .ids()
            .find(|id| controller.indices().contains(*id));
    };

    controller
        .probabilities()
        .ids()
        .skip_while(|&id| id != unit) // Probably cheaper than looking up in indices
        .find(|id| controller.indices().contains(*id))
}

/// Finds the units after the selected unit, and returns their ids and deltas.
fn sequential_unit_deltas<PR, TR>(
    update_buffer: &mut Vec<(PR::Id, PR::N)>,
    controller: &SampleController<PR, TR>,
    id: PR::Id,
    probability: PR::Value,
    quota: <PR as ProbabilityStore>::N,
) where
    PR: ProbabilityStore<N = f64>,
    TR: TreeStorage<PR::Id>,
{
    if controller.indices().is_empty() {
        return;
    }

    let mut remaining_weight: f64 = 1.0;

    for id_n in controller
        .probabilities()
        .ids()
        .skip_while(|&id_n| id_n != id)
        .skip(1) // Skip unit aswell
        .filter(|&id_n| controller.indices().contains(id_n))
    {
        let possible_weight = controller
            .probabilities()
            .weight_to(probability, id_n)
            .expect("id_n exists");
        let weight = possible_weight.min(remaining_weight);
        let delta = weight * quota;
        update_buffer.push((id_n, delta));
        remaining_weight -= possible_weight;
        if remaining_weight <= 0.0 {
            break;
        }
    }
}

/// Finds the neighbours of the selected unit, and returns their ids and deltas
#[inline]
fn spatial_unit_deltas<PR, DT>(
    update_buffer: &mut Vec<(PR::Id, PR::N)>,
    controller: &SampleController<PR, Tree<'_, DT>>,
    searcher: &mut WeightedSearcher<DT>,
    id: PR::Id,
    probability: Probability<f64>,
    quota: f64,
) where
    PR: ProbabilityStore<N = f64> + WeightCollection<PR::Id>,
    DT: PointSet<Id = PR::Id>,
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

    for n in searcher.neighbours().iter().take(guaranteed_units) {
        let delta = n.weight() * quota;
        update_buffer.push((*n.id(), delta));
        remaining_weight -= n.weight();
    }

    // Since the weights are sorted by distance, then weights, we'll try to remove weights by
    // removing as much as possible of the remaining shares, going from the smallest weight to
    // the largest.
    // It might be tempting to add a case before for when only one unit remains, but the only
    // thing we could save on below is a division by 1.0, as we don't know how much weight can
    // be used.
    let mut number_of_shares = (searcher.neighbours().len() - guaranteed_units)
        .to_f64()
        .expect("usize conv to f64");
    for n in searcher.neighbours().iter().skip(guaranteed_units) {
        let removable_weight = n.weight().min(remaining_weight / number_of_shares);
        let delta = remaining_weight * quota;
        update_buffer.push((*n.id(), delta));
        remaining_weight -= removable_weight;
        number_of_shares -= 1.0;
    }
}

/// Trait for selecting and deciding units based on random values
trait RandomUnit<PR, TR>
where
    PR: ProbabilityStore,
{
    /// Draws a unit. Defaults to a random unit among the indices.
    #[inline]
    #[must_use]
    fn select_unit<R>(
        &mut self,
        rng: &mut R,
        _id: Option<PR::Id>,
        controller: &SampleController<PR, TR>,
    ) -> Option<(PR::Id, PR::N)>
    where
        R: Rand<usize> + Rand<PR::N>,
    {
        controller.indices().draw(rng).map(|v| (v, rng.rand()))
    }
}

/// A strategy for CPS controls the updating and selection mechanism of units
trait CorrelatedPoissonStrategy<PR, TR>: RandomUnit<PR, TR>
where
    PR: ProbabilityStore,
    TR: TreeStorage<PR::Id>,
{
    /// Update the probabilities of affected units by the outcome of the selected unit `id`.
    /// `probability` refers to the probability of the selected unit, and `quota` of the
    /// probabilites to move (dependent on the outcome).
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, TR>,
        id: PR::Id,
        probability: PR::Value,
        quota: PR::N,
    );
}

/// Flip the coin for a unit. Returns (prob, quota)
#[inline]
fn decide_unit<PR, TR>(
    controller: &mut SampleController<PR, TR>,
    id: PR::Id,
    rv: PR::N,
) -> (PR::Value, PR::N)
where
    PR: ProbabilityStore,
    TR: TreeStorage<PR::Id>,
{
    let probability = controller
        .probabilities()
        .get(id)
        .copied()
        .expect("id to exist");
    let mut quota = probability.get();

    if rv < probability.get() {
        controller.unit_set_full(id);
        quota -= <PR::N as ConstOne>::ONE;
    } else {
        controller.unit_set_zero(id);
    }

    (probability, quota)
}

/// Runs a CPS strategy
#[inline]
fn correlated_poisson_runner<R, PR, TR, S>(
    rng: &mut R,
    mut controller: SampleController<PR, TR>,
    mut strategy: S,
) -> SampleController<PR, TR>
where
    R: Rand<PR::N>,
    PR: ProbabilityStore<N = f64>,
    TR: TreeStorage<PR::Id>,
    S: CorrelatedPoissonStrategy<PR, TR>,
{
    let Some((mut id, mut rv)) = strategy.select_unit(rng, None, &controller) else {
        return controller;
    };

    let mut updates = Vec::<(PR::Id, PR::N)>::with_capacity(controller.indices().len());

    loop {
        let (probability, quota) = decide_unit(&mut controller, id, rv);
        updates.clear();
        strategy.unit_deltas(&mut updates, &controller, id, probability, quota);
        for (id_n, delta) in &updates {
            let _prest = controller.unit_add_delta_and_decide(*id_n, *delta);
        }

        // Get next unit
        match strategy.select_unit(rng, Some(id), &controller) {
            Some((id_n, rv_n)) => {
                id = id_n;
                rv = rv_n;
            }
            None => {
                return controller;
            }
        }
    }
}

/// Sequential correlated Poisson sampling (or CPS)
struct SequentialStrategy;
impl<PR> RandomUnit<PR, ()> for SequentialStrategy
where
    PR: ProbabilityStore<N = f64>,
{
    fn select_unit<R>(
        &mut self,
        rng: &mut R,
        id: Option<PR::Id>,
        controller: &SampleController<PR, ()>,
    ) -> Option<(PR::Id, PR::N)>
    where
        R: Rand<usize> + Rand<PR::N>,
    {
        Some((select_unit_sequential(id, controller)?, rng.rand()))
    }
}
impl<PR> CorrelatedPoissonStrategy<PR, ()> for SequentialStrategy
where
    PR: ProbabilityStore<N = f64>,
{
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, ()>,
        id: PR::Id,
        probability: PR::Value,
        quota: PR::N,
    ) {
        sequential_unit_deltas(update_buffer, controller, id, probability, quota);
    }
}

/// Sequential correlated Poisson sampling (or CPS), coordinated variant
struct SequentialStrategyCoord<CD>(CD);
impl<PR, CD> RandomUnit<PR, ()> for SequentialStrategyCoord<CD>
where
    PR: ProbabilityStore<N = f64>,
    CD: DataView<Id = PR::Id, Value = f64>,
{
    fn select_unit<R>(
        &mut self,
        _rng: &mut R,
        id: Option<PR::Id>,
        controller: &SampleController<PR, ()>,
    ) -> Option<(<PR>::Id, PR::N)>
    where
        R: Rand<usize> + Rand<PR::N>,
    {
        let id = select_unit_sequential(id, controller)?;
        let rv = *self.0.get(id).expect("id to exist in coord");
        Some((id, rv))
    }
}
impl<PR, CD> CorrelatedPoissonStrategy<PR, ()> for SequentialStrategyCoord<CD>
where
    PR: ProbabilityStore<N = f64>,
    CD: DataView<Id = PR::Id, Value = f64>,
{
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, ()>,
        id: PR::Id,
        probability: PR::Value,
        quota: PR::N,
    ) {
        sequential_unit_deltas(update_buffer, controller, id, probability, quota);
    }
}

/// Spatially correlated Poisson sampling strategy
struct SpatialStrategy<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: WeightedSearcher<P>,
}
impl<PR, P> RandomUnit<PR, Tree<'_, P>> for SpatialStrategy<P>
where
    PR: ProbabilityStore<N = f64>,
    P: PointSet<Id = PR::Id>,
{
}
impl<PR, P> CorrelatedPoissonStrategy<PR, Tree<'_, P>> for SpatialStrategy<P>
where
    PR: ProbabilityStore<N = f64> + WeightCollection<PR::Id>,
    P: PointSet<Id = PR::Id>,
{
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, Tree<'_, P>>,
        id: <PR>::Id,
        probability: <PR>::Value,
        quota: <PR as ProbabilityStore>::N,
    ) {
        spatial_unit_deltas(
            update_buffer,
            controller,
            &mut self.searcher,
            id,
            probability,
            quota,
        );
    }
}

/// Coordinated Spatially correlated Poisson sampling strategy
struct SpatialStrategyCoord<P, CD>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: WeightedSearcher<P>,
    /// Random values
    random_values: CD,
}
impl<PR, P, CD> RandomUnit<PR, Tree<'_, P>> for SpatialStrategyCoord<P, CD>
where
    PR: ProbabilityStore<N = f64>,
    P: PointSet<Id = PR::Id>,
    CD: DataView<Id = PR::Id, Value = f64>,
{
    fn select_unit<R>(
        &mut self,
        _rng: &mut R,
        id: Option<<PR>::Id>,
        controller: &SampleController<PR, Tree<'_, P>>,
    ) -> Option<(<PR>::Id, <PR as ProbabilityStore>::N)>
    where
        R: Rand<usize> + Rand<<PR as ProbabilityStore>::N>,
    {
        let id = select_unit_sequential(id, controller)?;
        let rv = *self.random_values.get(id).expect("id to exist in coord");
        Some((id, rv))
    }
}
impl<PR, P, CD> CorrelatedPoissonStrategy<PR, Tree<'_, P>> for SpatialStrategyCoord<P, CD>
where
    PR: ProbabilityStore<N = f64> + WeightCollection<PR::Id>,
    P: PointSet<Id = PR::Id>,
    CD: DataView<Id = PR::Id, Value = f64>,
{
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, Tree<'_, P>>,
        id: <PR>::Id,
        probability: <PR>::Value,
        quota: <PR as ProbabilityStore>::N,
    ) {
        spatial_unit_deltas(
            update_buffer,
            controller,
            &mut self.searcher,
            id,
            probability,
            quota,
        );
    }
}

/// Locally correlated Poisson sampling
struct LocalStrategy<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: WeightedSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<P::Id>,
}
impl<PR, P> RandomUnit<PR, Tree<'_, P>> for LocalStrategy<P>
where
    PR: ProbabilityStore<N = f64> + WeightCollection<PR::Id>,
    P: PointSet<Id = PR::Id, Value: Number>,
{
    #[inline]
    fn select_unit<R>(
        &mut self,
        rng: &mut R,
        _id: Option<<PR>::Id>,
        controller: &SampleController<PR, Tree<'_, P>>,
    ) -> Option<(<PR>::Id, <PR as ProbabilityStore>::N)>
    where
        R: Rand<usize> + Rand<<PR as ProbabilityStore>::N>,
    {
        if controller.indices().len() <= 1 {
            return controller.indices().first().map(|id| (id, rng.rand()));
        } else if controller.indices().len() == 2 {
            return controller.indices().draw(rng).map(|id| (id, rng.rand()));
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
                    controller
                        .probabilities()
                        .get(id)
                        .expect("id to exist")
                        .get(),
                )
                .expect("id to exist")
                .search(controller.tree(), controller.probabilities())
                .expect("nn to be found");
            // We are guaranteed to have at least one neighbour by the if's in the beginning
            let distance = *self
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

        random_element(rng, &self.candidates)
            .copied()
            .map(|id| (id, rng.rand()))
    }
}
impl<PR, P> CorrelatedPoissonStrategy<PR, Tree<'_, P>> for LocalStrategy<P>
where
    PR: ProbabilityStore<N = f64> + WeightCollection<PR::Id>,
    P: PointSet<Id = PR::Id>,
{
    fn unit_deltas(
        &mut self,
        update_buffer: &mut Vec<(PR::Id, PR::N)>,
        controller: &SampleController<PR, Tree<'_, P>>,
        id: <PR>::Id,
        probability: <PR>::Value,
        quota: <PR as ProbabilityStore>::N,
    ) {
        spatial_unit_deltas(
            update_buffer,
            controller,
            &mut self.searcher,
            id,
            probability,
            quota,
        );
    }
}

/// Provides correlated Poisson sampling variants
pub trait CorrelatedPoissonSampling<ID, R>
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
    fn cps(&self, rng: &mut R) -> Vec<ID>;
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
    fn cps_coord<CD>(&self, rng: &mut R, random_values: CD) -> SamplingResult<Vec<ID>>
    where
        CD: DataView<Id = ID, Value = f64>;
}

/// Provides spatially correlated Poisson sampling variants
pub trait SpatiallyCorrelatedPoissonSampling<ID, R>
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
    fn scps(&self, rng: &mut R) -> Vec<ID>;
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
    fn scps_coord<CD>(&self, rng: &mut R, random_values: CD) -> SamplingResult<Vec<ID>>
    where
        CD: DataView<Id = ID, Value = f64>;
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
    fn lcps(&self, rng: &mut R) -> Vec<ID>;
}
impl<R, PO, AUX> CorrelatedPoissonSampling<PO::Id, R> for SamplingOptions<PO, AUX>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64>,
{
    #[inline]
    fn cps(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_real(self);
        correlated_poisson_runner(rng, controller, SequentialStrategy).to_sorted_sample_vec()
    }
    #[inline]
    fn cps_coord<CD>(&self, rng: &mut R, random_values: CD) -> SamplingResult<Vec<PO::Id>>
    where
        CD: DataView<Id = PO::Id, Value = f64>,
    {
        if !random_values.entries().all(|(id, rv)| {
            self.probabilities().contains(id) && Probability::is_real_probability(*rv)
        }) {
            return Err(SamplingError::InvalidRandomValues);
        }

        let controller = SampleController::new_real(self);
        let strategy = SequentialStrategyCoord(random_values);
        Ok(correlated_poisson_runner(rng, controller, strategy).to_sorted_sample_vec())
    }
}
impl<R, PO, P> SpatiallyCorrelatedPoissonSampling<PO::Id, R>
    for SamplingOptions<PO, SpreadingOptions<P>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id>,
{
    #[inline]
    fn scps(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_real_spreading(self);
        let searcher = WeightedSearcher::new(controller.tree().data());
        let strategy = SpatialStrategy { searcher };
        correlated_poisson_runner(rng, controller, strategy).to_sorted_sample_vec()
    }
    #[inline]
    fn scps_coord<CD>(&self, rng: &mut R, random_values: CD) -> SamplingResult<Vec<PO::Id>>
    where
        CD: DataView<Id = PO::Id, Value = f64>,
    {
        let controller = SampleController::new_real_spreading(self);
        let searcher = WeightedSearcher::new(controller.tree().data());
        let strategy = SpatialStrategyCoord {
            searcher,
            random_values,
        };
        Ok(correlated_poisson_runner(rng, controller, strategy).to_sorted_sample_vec())
    }
    #[inline]
    fn lcps(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_real_spreading(self);
        let searcher = WeightedSearcher::new(controller.tree().data());
        let strategy = LocalStrategy {
            searcher,
            candidates: Vec::<P::Id>::with_capacity(20),
        };
        correlated_poisson_runner(rng, controller, strategy).to_sorted_sample_vec()
    }
}

#[cfg(test)]
mod tests {
    use envisim_utils::probabilities::ProbabilityStoreToRaw;
    use envisim_utils::random::*;
    use envisim_utils::test_utils::*;

    use super::*;

    const RV_0: [f64; 10] = [0.0; 10];
    const RV_1: [f64; 10] = [1.0; 10];

    #[test]
    fn cps_sampler() -> SamplingResult<()> {
        let options = Data10::options_e();
        let mut con1 = SampleController::new_real(&options);
        let mut con2 = SampleController::new_real(&options);
        let p = Probability::new(0.2, con1.probabilities().ctx()).unwrap();

        assert_eq!(decide_unit(&mut con1, 7, 0.1), (p, -0.8));
        assert_eq!(decide_unit(&mut con2, 7, 0.3), (p, 0.2));
        Ok(())
    }

    fn decide_and_update<'a, S, PR, TR>(
        strategy: &mut S,
        controller: &mut SampleController<PR, TR>,
        id: PR::Id,
        rv: PR::N,
    ) -> (Probability<f64>, f64)
    where
        S: CorrelatedPoissonStrategy<PR, TR>,
        PR: ProbabilityStore<N = f64>,
        TR: TreeStorage<PR::Id>,
    {
        let mut updates = Vec::<(PR::Id, PR::N)>::new();
        let (p, q) = decide_unit(controller, id, rv);
        strategy.unit_deltas(&mut updates, &controller, id, p, q);
        for (id_n, delta) in &updates {
            let _prest = controller.unit_add_delta_and_decide(*id_n, *delta);
        }
        (p, q)
    }

    #[test]
    fn cps_variant() {
        let options = Data10::options_e();

        let mut con = SampleController::new_real(&options);
        let mut cpsv = SequentialStrategyCoord(RV_0);
        let _ = decide_and_update(&mut cpsv, &mut con, 0, 0.0);
        assert_vec!(con.probabilities().to_raw()[1..=4], vec![0.0; 4]);

        let mut con = SampleController::new_real(&options);
        let mut cpsv = SequentialStrategyCoord(RV_1);
        let _ = decide_and_update(&mut cpsv, &mut con, 0, 1.0);
        assert_vec!(con.probabilities().to_raw()[1..=4], vec![0.25; 4]);

        let mut rng = SmallRng::seed_from_u64(42);
        let s = options.cps(&mut rng);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn scps_variant() {
        let options = Data10::options_e();

        let mut con = SampleController::new_real_spreading(&options);
        let searcher = WeightedSearcher::new(con.tree().data());
        let mut cps = SpatialStrategy { searcher };
        let _ = decide_and_update(&mut cps, &mut con, 0, 0.0);
        assert_delta!(con.probabilities().get(1).unwrap().get(), 0.0);
        assert_delta!(con.probabilities().get(8).unwrap().get(), 0.0);
        assert_delta!(con.probabilities().get(4).unwrap().get(), 0.0);
        assert_delta!(con.probabilities().get(2).unwrap().get(), 0.0);

        let mut con = SampleController::new_real_spreading(&options);
        let searcher = WeightedSearcher::new(con.tree().data());
        let mut cps = SpatialStrategy { searcher };
        let _ = decide_and_update(&mut cps, &mut con, 9, 1.0);
        assert_delta!(con.probabilities().get(4).unwrap().get(), 0.25);
        assert_delta!(con.probabilities().get(2).unwrap().get(), 0.25);
        assert_delta!(con.probabilities().get(0).unwrap().get(), 0.25);
        assert_delta!(con.probabilities().get(7).unwrap().get(), 0.25);
    }

    #[test]
    fn lcps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let options = Data10::options_e();
        let con = SampleController::new_real_spreading(&options);
        let searcher = WeightedSearcher::new(con.tree().data());
        let mut cps = LocalStrategy {
            searcher,
            candidates: Vec::new(),
        };
        let (u, _) = cps.select_unit(&mut rng, None, &con).unwrap();
        assert_eq!(u, 8);
    }
}
