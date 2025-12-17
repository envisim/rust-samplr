// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
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

use envisim_utils::kd_tree::SearcherWeighted;
use envisim_utils::pips::{
    Probabilities,
    ProbabilitiesUnequal,
};
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::Enabled;
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};
use envisim_utils::utils::usize_to_f64;

use crate::sample_controller::{
    BasicSampleController,
    SampleController,
    SpreadingSampleController,
};

pub trait CorrelatedPoisson<C>
where
    C: SampleController<Probs = ProbabilitiesUnequal>,
{
    fn sample<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Vec<usize> {
        self.run(rng);
        self.controller_mut().sample_mut().sort_to_vec()
    }
    fn run<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        while let Some(id) = self.select_unit(rng) {
            let (p, q) = self.decide_unit(rng, id);
            self.update_probabilities(id, p, q);
        }
    }
    fn decide_unit<R: RandomNumberGenerator>(&mut self, rng: &mut R, id: usize) -> (f64, f64) {
        let probability = self.controller().probabilities().get(id);
        let mut quota = probability;

        if self.random_value(rng, id) < probability {
            self.controller_mut().unit_set_one(id);
            quota -= 1.0;
        } else {
            self.controller_mut().unit_set_zero(id);
        }

        (probability, quota)
    }
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    fn random_value<R: RandomNumberGenerator>(&mut self, rng: &mut R, id: usize) -> f64;
    fn select_unit<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Option<usize>;
    fn update_probabilities(&mut self, id: usize, probability: f64, quota: f64);
}

pub struct SequentialCorrelatedPoisson<'a> {
    controller: BasicSampleController<ProbabilitiesUnequal>,
    random_values: Option<&'a [f64]>,
    unit: usize,
}
impl<'a> CorrelatedPoisson<BasicSampleController<ProbabilitiesUnequal>>
    for SequentialCorrelatedPoisson<'a>
{
    fn controller(&self) -> &BasicSampleController<ProbabilitiesUnequal> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<ProbabilitiesUnequal> {
        &mut self.controller
    }
    fn random_value<R: RandomNumberGenerator>(&mut self, rng: &mut R, id: usize) -> f64 {
        self.random_values
            .map(|rv| rv[id])
            .unwrap_or_else(|| rng.rf64())
    }
    fn select_unit<R: RandomNumberGenerator>(&mut self, _: &mut R) -> Option<usize> {
        if self.controller.indices().is_empty() {
            return None;
        }

        let pop_size = self.controller.population_size();
        let unit = self.controller.indices().seq_after(self.unit, pop_size);

        if let Some(id) = unit {
            self.unit = id;
        }

        unit
    }
    fn update_probabilities(&mut self, id: usize, probability: f64, quota: f64) {
        if self.controller.indices().is_empty() {
            return;
        }

        let mut remaining_weight: f64 = 1.0;

        let pop_size = self.controller.population_size();
        let mut id_n: usize = id;
        while remaining_weight > 0.0 {
            id_n = match self.controller.indices().seq_after(id_n, pop_size) {
                Some(v) => v,
                None => return,
            };

            let possible_weight = self.controller.probabilities().weight_to(probability, id_n);
            let weight = possible_weight.min(remaining_weight);
            self.controller.unit_add_and_decide(id_n, weight * quota);
            remaining_weight -= possible_weight;
        }
    }
}
impl<'a> SequentialCorrelatedPoisson<'a> {
    pub fn new<P, S, B>(options: &'a SamplingOptions<'_, P, S, B>) -> Self
    where
        P: Probabilities,
        BasicSampleController<ProbabilitiesUnequal>: From<&'a SamplingOptions<'a, P, S, B>>,
    {
        Self {
            controller: options.into(),
            random_values: options.coordination().map(|c| c.data()),
            unit: 0,
        }
    }
}
/// Draw a sample using the (sequential) correlated poisson sampling method.
/// A variant of the cps where unit competes in order.
///
/// # Examples
/// ```
/// use envisim_samplr::correlated_poisson::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = SampleOptions::new(&p)?.sample(&mut rng, cps)?;
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
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let rv = [0.2; 10];
/// let opts = SampleOptions::new(&p)?.set_random_values(&rv)?;
/// let s = cps(&mut rng, &opts);
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
pub fn cps<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<ProbabilitiesUnequal>: From<&'a SamplingOptions<'a, P, S, B>>,
{
    SequentialCorrelatedPoisson::new(options).sample(rng)
}

pub struct SpatialCorrelatedPoisson<'a> {
    controller: SpreadingSampleController<'a, ProbabilitiesUnequal>,
    random_values: Option<(usize, &'a [f64])>,
    searcher: SearcherWeighted,
}
impl<'a> CorrelatedPoisson<SpreadingSampleController<'a, ProbabilitiesUnequal>>
    for SpatialCorrelatedPoisson<'a>
{
    fn controller(&self) -> &SpreadingSampleController<'a, ProbabilitiesUnequal> {
        &self.controller
    }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, ProbabilitiesUnequal> {
        &mut self.controller
    }
    fn random_value<R: RandomNumberGenerator>(&mut self, rng: &mut R, id: usize) -> f64 {
        self.random_values
            .map(|rv| rv.1[id])
            .unwrap_or_else(|| rng.rf64())
    }
    fn select_unit<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Option<usize> {
        if self.controller.indices().is_empty() {
            return None;
        }

        match self.random_values {
            Some((ref mut sunit, _)) => {
                let pop_size = self.controller.population_size();
                let unit = self.controller.indices().seq_after(*sunit, pop_size);

                if let Some(id) = unit {
                    *sunit = id;
                }

                unit
            }
            None => self.controller.indices().draw(rng),
        }
    }
    fn update_probabilities(&mut self, id: usize, probability: f64, quota: f64) {
        if self.controller.indices().is_empty() {
            return;
        }

        self.searcher
            .find_neighbours_of_iter(
                self.controller.tree(),
                self.controller.probabilities(),
                self.controller.tree().data().row_iter(id),
                probability,
            )
            .unwrap();

        let mut remaining_weight: f64 = 1.0;
        let mut i: usize = 0;

        while i < self.searcher.neighbours().len() {
            // Start by adding up the weights of all ties
            let mut sum_of_tie_weights = self.searcher.weight_k(i);
            let distance = self.searcher.distance_k(i);
            let mut j: usize = i + 1;

            while j < self.searcher.neighbours().len() && self.searcher.distance_k(j) == distance {
                sum_of_tie_weights += self.searcher.weight_k(j);
                j += 1;
            }

            // If the sum of all ties are less than the remaining weight, we can
            // continue as usual
            if sum_of_tie_weights < remaining_weight {
                while i < j {
                    let id = self.searcher.neighbours()[i];
                    let removable_weight = self.searcher.weight_k(i);
                    self.controller
                        .unit_add_and_decide(id, removable_weight * quota)
                        .expect("probability to be updated");
                    remaining_weight -= removable_weight;
                    i += 1;
                }

                i = j;
                continue;
            }

            // If the sum of all ties are more than the remaining weight, we need
            // to be a bit more tactful.
            // No unit should be able to get more than a "fair" share.
            // Initially, each unit should get equal weight.
            // If some units cannot accept this much weight, the remainder will
            // be redistributed amongst the others.
            // Thus, we sort the remaining neighbours with smallest first.
            let mut sharers = usize_to_f64(j - i);
            self.searcher.sort_by_weight(i, j);

            while i < j {
                let id = self.searcher.neighbours()[i];
                let removable_weight = self.searcher.weight_k(i).min(remaining_weight / sharers);
                self.controller
                    .unit_add_and_decide(id, removable_weight * quota);
                remaining_weight -= removable_weight;
                sharers -= 1.0;
                i += 1;
            }

            i = j;
        }
    }
}
impl<'a> SpatialCorrelatedPoisson<'a> {
    pub fn new<P, B>(options: &'a SamplingOptions<'_, P, Enabled, B>) -> Self
    where
        P: Probabilities,
        SpreadingSampleController<'a, ProbabilitiesUnequal>:
            From<&'a SamplingOptions<'a, P, Enabled, B>>,
    {
        let controller: SpreadingSampleController<'_, ProbabilitiesUnequal> = options.into();
        let searcher = SearcherWeighted::new(controller.tree());

        Self {
            controller,
            searcher,
            random_values: options.coordination().map(|c| (0, c.data())),
        }
    }
}
/// Draw a sample using the spatially correlated poisson sampling method.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::correlated_poisson::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let s = SampleOptions::new(&p)?.set_spreading(&m)?.sample(&mut rng, scps)?;
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
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let rv = [0.2; 10];
/// let opts = SampleOptions::new(&p)?
///     .set_spreading(&m)?
///     .set_random_values(&rv)?;
/// let s = scps(&mut rng, &opts);
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
pub fn scps<R, P, B>(rng: &mut R, options: &SamplingOptions<'_, P, Enabled, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    SpatialCorrelatedPoisson::new(options).sample(rng)
}

pub struct LocalCorrelatedPoisson<'a> {
    scps: SpatialCorrelatedPoisson<'a>,
    candidates: Vec<usize>,
}
impl<'a> CorrelatedPoisson<SpreadingSampleController<'a, ProbabilitiesUnequal>>
    for LocalCorrelatedPoisson<'a>
{
    fn controller(&self) -> &SpreadingSampleController<'a, ProbabilitiesUnequal> {
        self.scps.controller()
    }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, ProbabilitiesUnequal> {
        self.scps.controller_mut()
    }
    fn random_value<R: RandomNumberGenerator>(&mut self, rng: &mut R, _: usize) -> f64 {
        rng.rf64()
    }
    fn select_unit<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Option<usize> {
        if self.controller().indices().len() <= 1 {
            return self.controller().indices().first();
        } else if self.controller().indices().len() == 2 {
            return self.controller().indices().draw(rng);
        }

        let mut minimum_distance = f64::MAX;
        self.candidates.clear();

        // Loop through all remaining units
        let mut i = 0;
        while i < self.controller().indices().len() {
            let id = self.controller().indices().get(i).unwrap();
            self.scps
                .searcher
                .find_neighbours_of_id(
                    self.scps.controller.tree(),
                    self.scps.controller.probabilities(),
                    id,
                )
                .unwrap();
            // We are guaranteed to have at least one neighbour by the
            // if's in the beginning
            let distance = self
                .scps
                .searcher
                .distance_k(self.scps.searcher.neighbours().len() - 1);

            if distance < minimum_distance {
                self.candidates.clear();
                self.candidates.push(id);
                minimum_distance = distance;
            } else if distance == minimum_distance {
                self.candidates.push(id);
            }

            i += 1;
        }

        rng.relement(&self.candidates).cloned()
    }
    fn update_probabilities(&mut self, id: usize, probability: f64, quota: f64) {
        self.scps.update_probabilities(id, probability, quota);
    }
}
impl<'a> LocalCorrelatedPoisson<'a> {
    pub fn new<P, B>(options: &'a SamplingOptions<'_, P, Enabled, B>) -> Self
    where
        P: Probabilities,
        SpreadingSampleController<'a, ProbabilitiesUnequal>:
            From<&'a SamplingOptions<'a, P, Enabled, B>>,
    {
        let controller: SpreadingSampleController<'_, ProbabilitiesUnequal> = options.into();
        let searcher = SearcherWeighted::new(controller.tree());

        Self {
            scps: SpatialCorrelatedPoisson {
                controller,
                searcher,
                random_values: None,
            },
            candidates: Vec::<usize>::with_capacity(20),
        }
    }
}
/// Draw a sample using the locally correlated poisson sampling method.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::correlated_poisson::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let opts = SampleOptions::new(&p)?.set_spreading(&m);
/// let s = lcps(&mut rng, &opts);
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
pub fn lcps<R, P, B>(rng: &mut R, options: &SamplingOptions<'_, P, Enabled, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    LocalCorrelatedPoisson::new(options).sample(rng)
}

#[cfg(test)]
mod tests {
    use envisim_test_utils::*;
    use envisim_utils::matrix::Matrix;
    use envisim_utils::random::*;
    use envisim_utils::sampling_options::{
        Disabled,
        SamplingOptionsError,
    };

    use super::*;

    const RV_0: [f64; 10] = [0.0; 10];
    const RV_1: [f64; 10] = [1.0; 10];

    fn options() -> SamplingOptions<'static, ProbabilitiesUnequal, Disabled, Disabled> {
        SamplingOptions::new(&PROB_10_E).unwrap()
    }
    fn options_coord(
        zero: bool,
    ) -> SamplingOptions<'static, ProbabilitiesUnequal, Disabled, Disabled> {
        options()
            .set_coordination(if zero { &RV_0 } else { &RV_1 })
            .unwrap()
    }

    #[test]
    fn cps_sampler() -> Result<(), SamplingOptionsError> {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = options_coord(true);
        let mut cps = SequentialCorrelatedPoisson::new(&options);
        assert_eq!(cps.decide_unit(&mut rng, 7), (0.2, -0.8));

        let options = options_coord(false);
        let mut cps = SequentialCorrelatedPoisson::new(&options);
        assert_eq!(cps.decide_unit(&mut rng, 7), (0.2, 0.2));
        Ok(())
    }

    fn decide_and_update<'a, R, C, S>(cps: &mut C, rng: &mut R, id: usize) -> (f64, f64)
    where
        R: RandomNumberGenerator,
        C: CorrelatedPoisson<S>,
        S: SampleController<Probs = ProbabilitiesUnequal>,
    {
        let (p, q) = cps.decide_unit(rng, id);
        cps.update_probabilities(id, p, q);
        (p, q)
    }

    #[test]
    fn cps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);

        let options = options_coord(true);
        let mut cps = SequentialCorrelatedPoisson::new(&options);
        decide_and_update(&mut cps, &mut rng, 0);
        assert_fvec(&cps.controller.probabilities().data()[1..=4], &vec![0.0; 4]);

        let options = options_coord(false);
        let mut cps = SequentialCorrelatedPoisson::new(&options);
        decide_and_update(&mut cps, &mut rng, 0);
        assert_fvec(
            &cps.controller.probabilities().data()[1..=4],
            &vec![0.25; 4],
        );
    }

    #[test]
    fn scps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let data = Matrix::new(&DATA_10_2, 10).unwrap();

        let options = options_coord(true).set_spreading(&data).unwrap();
        let mut cps = SpatialCorrelatedPoisson::new(&options);
        decide_and_update(&mut cps, &mut rng, 0);
        assert_delta!(cps.controller.probabilities()[1], 0.0);
        assert_delta!(cps.controller.probabilities()[8], 0.0);
        assert_delta!(cps.controller.probabilities()[4], 0.0);
        assert_delta!(cps.controller.probabilities()[2], 0.0);

        let options = options_coord(false).set_spreading(&data).unwrap();
        let mut cps = SpatialCorrelatedPoisson::new(&options);
        decide_and_update(&mut cps, &mut rng, 9);
        assert_delta!(cps.controller.probabilities()[4], 0.25);
        assert_delta!(cps.controller.probabilities()[2], 0.25);
        assert_delta!(cps.controller.probabilities()[0], 0.25);
        assert_delta!(cps.controller.probabilities()[7], 0.25);
    }

    #[test]
    fn lcps_variant() {
        let mut rng = SmallRng::seed_from_u64(42);
        let data = Matrix::new(&DATA_10_2, 10).unwrap();

        let options = options_coord(true).set_spreading(&data).unwrap();
        let mut cps = LocalCorrelatedPoisson::new(&options);
        assert_eq!(cps.select_unit(&mut rng), Some(8));
        decide_and_update(&mut cps, &mut rng, 8);
        assert_delta!(cps.controller().probabilities()[3], 0.0, EPS);
        assert_delta!(cps.controller().probabilities()[5], 0.0, EPS);
        assert_delta!(cps.controller().probabilities()[2], 0.0, EPS);
        assert_delta!(cps.controller().probabilities()[1], 0.0, EPS);

        let options = options_coord(false).set_spreading(&data).unwrap();
        let mut cps = LocalCorrelatedPoisson::new(&options);
        decide_and_update(&mut cps, &mut rng, 8);
        assert_delta!(cps.controller().probabilities()[3], 0.25, EPS);
        assert_delta!(cps.controller().probabilities()[5], 0.25, EPS);
        assert_delta!(cps.controller().probabilities()[2], 0.25, EPS);
        assert_delta!(cps.controller().probabilities()[1], 0.25, EPS);
    }
}
