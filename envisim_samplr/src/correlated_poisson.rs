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

use envisim_utils::kd_tree::searcher::WeightedSearcher;
use envisim_utils::probabilities::{
    FloatProbabilities,
    ProbabilityStore,
};
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::{
    BasicSampleController,
    SampleController,
    SpreadingSampleController,
};
use envisim_utils::sampling_options::ProbabilitySpec;
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
    SamplingOptionsResult,
};
use envisim_utils::spatial::{
    Number,
    PointSet,
};
use envisim_utils::utils::usize_to_f64;

pub use crate::error::SamplingError;
use crate::error::SamplingResult;

pub trait CorrelatedPoissonStrategy<C> {
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: RandomNumberGenerator;
    fn select_unit<R>(&mut self, controller: &mut C, rng: &mut R) -> Option<usize>
    where
        C: SampleController<Store = FloatProbabilities>,
        R: RandomNumberGenerator;
    fn update_probabilities(&mut self, controller: &mut C, id: usize, probability: f64, quota: f64)
    where
        C: SampleController<Store = FloatProbabilities>;
}

pub struct CorrelatedPoissonRunner<C, S> {
    controller: C,
    strategy: S,
}

impl<C, S> CorrelatedPoissonRunner<C, S>
where
    C: SampleController<Store = FloatProbabilities>,
    S: CorrelatedPoissonStrategy<C>,
{
    fn sample<R>(&mut self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        self.run(rng);
        self.controller.sample_mut().sort_to_vec()
    }
    fn run<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        while let Some(id) = self.strategy.select_unit(&mut self.controller, rng) {
            let (p, q) = self.decide_unit(rng, id);
            self.strategy
                .update_probabilities(&mut self.controller, id, p, q);
        }
    }
    fn decide_unit<R>(&mut self, rng: &mut R, id: usize) -> (f64, f64)
    where
        R: RandomNumberGenerator,
    {
        let probability = self.controller.probabilities().get(id);
        let mut quota = probability;
        let rv = self.strategy.random_value(rng, id);

        if rv < probability {
            self.controller.unit_set_max(id);
            quota -= 1.0;
        } else {
            self.controller.unit_set_zero(id);
        }

        (probability, quota)
    }
}

pub struct SequentialStrategy<'a> {
    random_values: Option<&'a [f64]>,
    unit: usize,
}
impl<'a> SequentialStrategy<'a> {
    pub fn new<PS, SOP, M>(
        options: &'a SamplingOptions<'_, PS, SOP, M>,
    ) -> CorrelatedPoissonRunner<BasicSampleController<FloatProbabilities>, Self>
    where
        PS: ProbabilitySpec,
    {
        let controller = options.to_controller_float();
        CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values: options.coordination().map(|c| c.data()),
                unit: 0,
            },
        }
    }
}
impl<C> CorrelatedPoissonStrategy<C> for SequentialStrategy<'_>
where
    C: SampleController<Store = FloatProbabilities>,
{
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: RandomNumberGenerator,
    {
        self.random_values
            .map(|rv| rv[id])
            .unwrap_or_else(|| rng.rf64())
    }
    fn select_unit<R>(&mut self, controller: &mut C, _rng: &mut R) -> Option<usize>
    where
        R: RandomNumberGenerator,
    {
        if controller.indices().is_empty() {
            return None;
        } else if self.unit == 0 && controller.indices().contains(0) {
            // Special case for 0, as it is set 0 at construction
            return Some(0);
        }

        let pop_size = controller.population_size();
        let unit = controller.indices().seq_after(self.unit, pop_size);

        if let Some(id) = unit {
            self.unit = id;
        }

        unit
    }
    fn update_probabilities(
        &mut self,
        controller: &mut C,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        if controller.indices().is_empty() {
            return;
        }

        let mut remaining_weight: f64 = 1.0;

        let pop_size = controller.population_size();
        let mut id_n: usize = id;
        while remaining_weight > 0.0 {
            id_n = match controller.indices().seq_after(id_n, pop_size) {
                Some(v) => v,
                None => return,
            };

            let possible_weight = controller.probabilities().weight_to(probability, id_n);
            let weight = possible_weight.min(remaining_weight);
            controller.unit_add_and_decide(id_n, weight * quota);
            remaining_weight -= possible_weight;
        }
    }
}

pub struct SpatialStrategy<'a, N> {
    random_values: Option<(usize, &'a [f64])>,
    searcher: WeightedSearcher<N>,
}
impl<'b, N> SpatialStrategy<'b, N> {
    pub fn new<PS, SOP, M>(
        options: &'b SamplingOptions<'_, PS, SOP, M>,
    ) -> SamplingResult<
        CorrelatedPoissonRunner<
            SpreadingSampleController<'b, FloatProbabilities, N, SOP>,
            SpatialStrategy<'b, N>,
        >,
    >
    where
        N: Number,
        PS: ProbabilitySpec,
        SOP: PointSet<N>,
    {
        let controller = options.to_spreading_controller_float()?;
        let searcher = WeightedSearcher::new(controller.tree().data());
        Ok(CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                random_values: options.coordination().map(|c| (0, c.data())),
                searcher,
            },
        })
    }
}
fn spatial_update_probabilities<P, N>(
    searcher: &mut WeightedSearcher<N>,
    controller: &mut SpreadingSampleController<'_, FloatProbabilities, N, P>,
    id: usize,
    probability: f64,
    quota: f64,
) where
    P: PointSet<N>,
    N: Number,
{
    if controller.indices().is_empty() {
        return;
    }

    searcher
        .reset_from_unit(controller.tree().data(), id, probability)
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

    for n in searcher.neighbours()[0..guaranteed_units].iter() {
        controller
            .unit_add_and_decide(n.id(), n.weight() * quota)
            .expect("probability to be updated");
        remaining_weight -= n.weight();
    }

    let sum_of_tie_weights: f64 = searcher.neighbours()[guaranteed_units..]
        .iter()
        .map(|n| n.weight())
        .sum();

    if sum_of_tie_weights == remaining_weight {
        // Add everything left, if it's exactly solved (unlikely)
        for n in searcher.neighbours()[guaranteed_units..].iter() {
            controller
                .unit_add_and_decide(n.id(), n.weight() * quota)
                .expect("probability to be updated");
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
    let mut number_of_shares = usize_to_f64(searcher.neighbours().len() - guaranteed_units);
    for n in searcher.neighbours()[guaranteed_units..].iter() {
        let removable_weight = n.weight().min(remaining_weight / number_of_shares);
        controller
            .unit_add_and_decide(n.id(), remaining_weight * quota)
            .expect("probability to be updated");
        remaining_weight -= removable_weight;
        number_of_shares -= 1.0;
    }
}

impl<'a, P, N> CorrelatedPoissonStrategy<SpreadingSampleController<'a, FloatProbabilities, N, P>>
    for SpatialStrategy<'a, N>
where
    P: PointSet<N>,
    N: Number,
{
    fn random_value<R>(&mut self, rng: &mut R, id: usize) -> f64
    where
        R: RandomNumberGenerator,
    {
        self.random_values
            .map(|rv| rv.1[id])
            .unwrap_or_else(|| rng.rf64())
    }
    fn select_unit<R>(
        &mut self,
        controller: &mut SpreadingSampleController<'a, FloatProbabilities, N, P>,
        rng: &mut R,
    ) -> Option<usize>
    where
        R: RandomNumberGenerator,
    {
        if controller.indices().is_empty() {
            return None;
        }

        match self.random_values {
            Some((ref mut sunit, _)) => {
                let pop_size = controller.population_size();
                let unit = controller.indices().seq_after(*sunit, pop_size);

                if let Some(id) = unit {
                    *sunit = id;
                }

                unit
            }
            None => controller.indices().draw(rng),
        }
    }
    fn update_probabilities(
        &mut self,
        controller: &mut SpreadingSampleController<'a, FloatProbabilities, N, P>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        spatial_update_probabilities(&mut self.searcher, controller, id, probability, quota);
    }
}

pub struct LocalStrategy<N> {
    searcher: WeightedSearcher<N>,
    candidates: Vec<usize>,
}
impl<N> LocalStrategy<N> {
    pub fn new<'b, PS, SOP, M>(
        options: &'b SamplingOptions<'_, PS, SOP, M>,
    ) -> SamplingResult<
        CorrelatedPoissonRunner<
            SpreadingSampleController<'b, FloatProbabilities, N, SOP>,
            LocalStrategy<N>,
        >,
    >
    where
        N: Number,
        PS: ProbabilitySpec,
        SOP: PointSet<N>,
    {
        let controller = options.to_spreading_controller_float()?;
        let searcher = WeightedSearcher::new(controller.tree().data());
        Ok(CorrelatedPoissonRunner {
            controller,
            strategy: Self {
                searcher,
                candidates: Vec::<usize>::with_capacity(20),
            },
        })
    }
}
impl<N, P> CorrelatedPoissonStrategy<SpreadingSampleController<'_, FloatProbabilities, N, P>>
    for LocalStrategy<N>
where
    P: PointSet<N>,
    N: Number,
{
    fn random_value<R>(&mut self, rng: &mut R, _id: usize) -> f64
    where
        R: RandomNumberGenerator,
    {
        rng.rf64()
    }
    fn select_unit<R>(
        &mut self,
        controller: &mut SpreadingSampleController<'_, FloatProbabilities, N, P>,
        rng: &mut R,
    ) -> Option<usize>
    where
        R: RandomNumberGenerator,
    {
        if controller.indices().len() <= 1 {
            return controller.indices().first();
        } else if controller.indices().len() == 2 {
            return controller.indices().draw(rng);
        }

        let mut minimum_distance = N::max_value();
        self.candidates.clear();

        // Loop through all remaining units
        let mut i = 0;
        while i < controller.indices().len() {
            let id = controller.indices().get(i).unwrap();
            self.searcher
                .reset_from_unit(
                    controller.tree().data(),
                    id,
                    controller.probabilities().get(id),
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

        rng.relement(&self.candidates).cloned()
    }
    fn update_probabilities(
        &mut self,
        controller: &mut SpreadingSampleController<'_, FloatProbabilities, N, P>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        spatial_update_probabilities(&mut self.searcher, controller, id, probability, quota);
    }
}

pub trait CorrelatedPoissonSampling {
    fn cps<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator;
}
pub trait SpatiallyCorrelatedPoissonSampling<P, N>
where
    P: PointSet<N>,
    N: Number,
{
    fn scps<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
    fn lcps<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
}
impl<PS, SOP, M> CorrelatedPoissonSampling for SamplingOptions<'_, PS, SOP, M>
where
    PS: ProbabilitySpec,
{
    /// Draw a sample using the (sequential) correlated poisson sampling method.
    /// A variant of the cps where unit competes in order.
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let opts = SamplingOptions::new(&p)?;
    /// let s = opts.cps(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// ## Coordination
    /// `random_values` are used in order to decide the inclusions of units, allowing for coordination
    /// between multiple sampling efforts.
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let rv = [0.2; 10];
    /// let opts = SamplingOptions::new(&p)?.set_coordination(&rv)?;
    /// let s = opts.cps(&mut rng);
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Bondesson, L., & Thorburn, D. (2008).
    /// A list sequential sampling method suitable for real‐time sampling.
    /// Scandinavian Journal of Statistics, 35(3), 466-483.
    /// <https://doi.org/10.1111/j.1467-9469.2008.00596.x>
    fn cps<R>(&self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        SequentialStrategy::new(self).sample(rng)
    }
}
impl<PS, SOP, N, M> SpatiallyCorrelatedPoissonSampling<SOP, N> for SamplingOptions<'_, PS, SOP, M>
where
    PS: ProbabilitySpec,
    SOP: PointSet<N>,
    N: Number,
{
    /// Draw a sample using the spatially correlated poisson sampling method.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    /// use envisim_utils::matrix::Matrix;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let opts = SamplingOptions::new(&p)?.set_spreading(m)?;
    /// let s = opts.scps(&mut rng)?;
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// ## Coordination
    /// `random_values` are used in order to decide the inclusions of units, allowing for coordination
    /// between multiple sampling efforts.
    /// ```
    /// use envisim_samplr::correlated_poisson::*;
    /// use envisim_utils::random::*;
    /// use envisim_utils::matrix::Matrix;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let rv = [0.2; 10];
    /// let opts = SamplingOptions::new(&p)?
    ///     .set_spreading(m)?
    ///     .set_coordination(&rv)?;
    /// let s = opts.scps(&mut rng)?;
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A. (2012).
    /// Spatially correlated Poisson sampling.
    /// Journal of Statistical Planning and Inference, 142(1), 139-147.
    /// <https://doi.org/10.1016/j.jspi.2011.07.003>
    fn scps<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(SpatialStrategy::new(self)?.sample(rng))
    }
    /// Draw a sample using the locally correlated poisson sampling method.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    /// use envisim_utils::matrix::Matrix;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let opts = SamplingOptions::new(&p)?.set_spreading(m)?;
    /// let s = opts.lcps(&mut rng)?;
    ///
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingOptionsError>(())
    /// ```
    ///
    /// # References
    /// Prentius, W. (2024).
    /// Locally correlated Poisson sampling.
    /// Environmetrics, 35(2), e2832.
    /// <https://doi.org/10.1002/env.2832>
    fn lcps<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(LocalStrategy::new(self)?.sample(rng))
    }
}

#[cfg(test)]
mod tests {
    use envisim_test_utils::*;
    use envisim_utils::matrix::Matrix;
    use envisim_utils::random::*;
    use envisim_utils::sampling_options::{
        ProbabilitySpecEqual,
        ProbabilitySpecUnequal,
        SamplingOptionsError,
    };

    use super::*;

    const RV_0: [f64; 10] = [0.0; 10];
    const RV_1: [f64; 10] = [1.0; 10];

    fn options_ue() -> SamplingOptions<'static, ProbabilitySpecUnequal<'static>, (), ()> {
        SamplingOptions::new(&PROB_10_E).unwrap()
    }
    fn options_coord(
        zero: bool,
    ) -> SamplingOptions<'static, ProbabilitySpecUnequal<'static>, (), ()> {
        options_ue()
            .set_coordination(if zero { &RV_0 } else { &RV_1 })
            .unwrap()
    }

    #[test]
    fn cps_sampler() -> Result<(), SamplingOptionsError> {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = options_coord(true);
        let mut cps = options.to_cps();
        assert_eq!(cps.decide_unit(&mut rng, 7), (0.2, -0.8));

        let options = options_coord(false);
        let mut cps = options.to_cps();
        assert_eq!(cps.decide_unit(&mut rng, 7), (0.2, 0.2));
        Ok(())
    }

    fn decide_and_update<'a, R, C, S>(cps: &mut C, rng: &mut R, id: usize) -> (f64, f64)
    where
        R: RandomNumberGenerator,
        C: CorrelatedPoisson<S>,
        S: SampleController<Store = FloatProbabilities>,
    {
        let (p, q) = cps.decide_unit(rng, id);
        cps.update_probabilities(id, p, q);
        (p, q)
    }

    #[test]
    fn cps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = options_coord(true);
        let mut cpsv = options.to_cps();
        decide_and_update(&mut cpsv, &mut rng, 0);
        assert_fvec(
            &cpsv.controller.probabilities().data()[1..=4],
            &vec![0.0; 4],
        );

        let options = options_coord(false);
        let mut cpsv = options.to_cps();
        decide_and_update(&mut cpsv, &mut rng, 0);
        assert_fvec(
            &cpsv.controller.probabilities().data()[1..=4],
            &vec![0.25; 4],
        );

        // let options = options_ue();
        println!("CPS1");
        let mut rng = SmallRng::seed_from_u64(42);
        let options = options_ue();
        let s = options.cps(&mut rng);
        assert_eq!(s.len(), 2);
        println!("CPS2");
        // let mut rng = SmallRng::seed_from_u64(42);
        let options: SamplingOptions<'static, ProbabilitySpecEqual> = (10, 2).try_into().unwrap();
        let s = options.cps(&mut rng);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn scps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let data = Matrix::new(&DATA_10_2, 10).unwrap();

        let options = options_coord(true)
            .set_spreading(data.clone_shallow())
            .unwrap();
        let mut cps = options.to_scps().unwrap();
        decide_and_update(&mut cps, &mut rng, 0);
        println!("{:?}", cps.controller.probabilities().data());
        assert_delta!(cps.controller.probabilities().get(1), 0.0);
        assert_delta!(cps.controller.probabilities().get(8), 0.0);
        assert_delta!(cps.controller.probabilities().get(4), 0.0);
        assert_delta!(cps.controller.probabilities().get(2), 0.0);

        let options = options_coord(false)
            .set_spreading(data.clone_shallow())
            .unwrap();
        let mut cps = options.to_scps().unwrap();
        decide_and_update(&mut cps, &mut rng, 9);
        assert_delta!(cps.controller.probabilities().get(4), 0.25);
        assert_delta!(cps.controller.probabilities().get(2), 0.25);
        assert_delta!(cps.controller.probabilities().get(0), 0.25);
        assert_delta!(cps.controller.probabilities().get(7), 0.25);
    }

    #[test]
    fn lcps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let data = Matrix::new(&DATA_10_2, 10).unwrap();

        let options = options_coord(true).set_spreading(data).unwrap();
        let mut cps = options.to_lcps().unwrap();
        assert_eq!(cps.select_unit(&mut rng), Some(8));
    }
}
