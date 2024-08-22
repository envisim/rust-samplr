// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

use crate::utils::Container;
use crate::SampleOptions;
use envisim_utils::error::SamplingError;
use envisim_utils::kd_tree::{Node, SearcherWeighted};
use envisim_utils::matrix::OperateMatrix;
use envisim_utils::utils::{random_element, usize_to_f64};
use rand::Rng;

pub trait CorrelatedPoissonVariant<'a, R>
where
    R: Rng + ?Sized,
{
    fn select_unit(&mut self, container: &mut Container<'a, R>) -> Option<usize>;
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    );
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct CorrelatedPoissonSampler<'a, R, T>
where
    R: Rng + ?Sized,
    T: CorrelatedPoissonVariant<'a, R>,
{
    container: Box<Container<'a, R>>,
    variant: Box<T>,
    random_values: Option<&'a [f64]>,
}

pub struct SequentialCorrelatedPoissonSampling {
    unit: usize,
}

pub struct SpatiallyCorrelatedPoissonSampling<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<SearcherWeighted>,
    unit: Option<usize>, // Sequential also, usize::MAX
}

pub struct LocallyCorrelatedPoissonSampling<'a> {
    scps: SpatiallyCorrelatedPoissonSampling<'a>,
    candidates: Vec<usize>,
}

/// Draw a sample using the (sequential) correlated poisson sampling method.
/// A variant of the cps where unit competes in order.
///
/// # Examples
/// ```
/// use envisim_samplr::poisson::*;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = SampleOptions::new(&p)?.sample(&mut rng, cps)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// ## Coordination
/// `random_values` are used in order to decide the inclusions of units, allowing for coordination
/// between multiple sampling efforts.
/// ```
/// use envisim_samplr::poisson::*;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let rv = [0.2; 10];
/// let s = SampleOptions::new(&p)?.random_values(&rv)?.sample(&mut rng, cps)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Bondesson, L., & Thorburn, D. (2008).
/// A list sequential sampling method suitable for real‐time sampling.
/// Scandinavian Journal of Statistics, 35(3), 466-483.
/// <https://doi.org/10.1111/j.1467-9469.2008.00596.x>
#[inline]
pub fn cps<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    cps_new(rng, options)?.sample_with_return()
}
#[inline]
fn cps_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<CorrelatedPoissonSampler<'a, R, SequentialCorrelatedPoissonSampling>, SamplingError>
where
    R: Rng + ?Sized,
{
    Ok(CorrelatedPoissonSampler {
        container: Container::new_boxed(rng, options)?,
        variant: Box::new(SequentialCorrelatedPoissonSampling { unit: 0 }),
        random_values: options.random_values,
    })
}

/// Draw a sample using the spatially correlated poisson sampling method.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::poisson::*;
/// use envisim_utils::matrix::RefMatrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let dt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// let m = RefMatrix::new(&dt, 10);
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.sample(&mut rng, scps)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// ## Coordination
/// `random_values` are used in order to decide the inclusions of units, allowing for coordination
/// between multiple sampling efforts.
/// ```
/// use envisim_samplr::poisson::*;
/// use envisim_utils::matrix::RefMatrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let dt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// let m = RefMatrix::new(&dt, 10);
/// let rv = [0.2; 10];
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.random_values(&rv)?.sample(&mut rng, scps)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A. (2012).
/// Spatially correlated Poisson sampling.
/// Journal of Statistical Planning and Inference, 142(1), 139-147.
/// <https://doi.org/10.1016/j.jspi.2011.07.003>
#[inline]
pub fn scps<'a, R>(rng: &'a mut R, options: &SampleOptions<'a>) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    scps_new(rng, options)?.sample_with_return()
}
#[inline]
fn scps_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<CorrelatedPoissonSampler<'a, R, SpatiallyCorrelatedPoissonSampling<'a>>, SamplingError>
where
    R: Rng + ?Sized,
{
    options.check_spatially_balanced()?;
    let container = Container::new_boxed(rng, options)?;
    let tree = options.build_node(&mut container.indices().to_vec())?;
    let searcher = Box::new(SearcherWeighted::new(&tree)?);

    Ok(CorrelatedPoissonSampler {
        container,
        variant: Box::new(SpatiallyCorrelatedPoissonSampling {
            tree,
            searcher,
            unit: options.random_values.and(Some(0)),
        }),
        random_values: None,
    })
}

/// Draw a sample using the locally correlated poisson sampling method.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::poisson::*;
/// use envisim_utils::matrix::RefMatrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let dt = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// let m = RefMatrix::new(&dt, 10);
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.sample(&mut rng, lcps)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Prentius, W. (2024).
/// Locally correlated Poisson sampling.
/// Environmetrics, 35(2), e2832.
/// <https://doi.org/10.1002/env.2832>
#[inline]
pub fn lcps<'a, R>(rng: &'a mut R, options: &SampleOptions<'a>) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    lcps_new(rng, options)?.sample_with_return()
}
#[inline]
fn lcps_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<CorrelatedPoissonSampler<'a, R, LocallyCorrelatedPoissonSampling<'a>>, SamplingError>
where
    R: Rng + ?Sized,
{
    options.check_spatially_balanced()?;
    let container = Container::new_boxed(rng, options)?;
    let tree = options.build_node(&mut container.indices().to_vec())?;
    let searcher = Box::new(SearcherWeighted::new(&tree)?);

    Ok(CorrelatedPoissonSampler {
        container,
        variant: Box::new(LocallyCorrelatedPoissonSampling {
            scps: SpatiallyCorrelatedPoissonSampling {
                tree,
                searcher,
                unit: None,
            },
            candidates: Vec::<usize>::with_capacity(20),
        }),
        random_values: None,
    })
}

impl<'a, R, T> CorrelatedPoissonSampler<'a, R, T>
where
    R: Rng + ?Sized,
    T: CorrelatedPoissonVariant<'a, R>,
{
    #[inline]
    fn decide_selected(&mut self, id: usize, rv: f64) -> (f64, f64) {
        let probability = self.container.probabilities()[id];
        let mut quota = probability;

        if rv < probability {
            self.container.probabilities_mut()[id] = 1.0;
            quota -= 1.0;
        } else {
            self.container.probabilities_mut()[id] = 0.0;
        }

        self.variant.decide_unit(&mut self.container, id);

        (probability, quota)
    }
    #[inline]
    fn sample_with_return(&mut self) -> Result<Vec<usize>, SamplingError> {
        Ok(self.sample().get_sorted_sample().to_vec())
    }
    #[inline]
    fn sample(&mut self) -> &mut Self {
        while let Some(id) = self.variant.select_unit(&mut self.container) {
            let rv: f64 = match self.random_values {
                Some(list) => list[id],
                None => self.container.rng().gen::<f64>(),
            };

            let (probability, quota) = self.decide_selected(id, rv);
            self.variant
                .update_neighbours(&mut self.container, id, probability, quota);
        }

        self
    }
    #[inline]
    fn get_sorted_sample(&mut self) -> &[usize] {
        self.container.sample_mut().sort().get()
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for SequentialCorrelatedPoissonSampling
where
    R: Rng + ?Sized,
{
    fn select_unit(&mut self, container: &mut Container<'a, R>) -> Option<usize> {
        if container.indices().is_empty() {
            return None;
        }

        while self.unit < container.population_size() {
            if container.indices().contains(self.unit) {
                return Some(self.unit);
            }

            self.unit += 1;
        }

        unsafe { std::hint::unreachable_unchecked() };
    }
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        let mut nid = id + 1;
        let mut remaining_weight: f64 = 1.0;

        while nid < container.population_size() && remaining_weight > 0.0 {
            if !container.indices().contains(nid) {
                nid += 1;
                continue;
            }

            let possible_weight = container.probabilities().weight_to(probability, nid);
            let weight = possible_weight.min(remaining_weight);
            container.probabilities_mut()[nid] += weight * quota;
            self.decide_unit(container, nid);
            remaining_weight -= possible_weight;
            nid += 1;
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap()
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for SpatiallyCorrelatedPoissonSampling<'a>
where
    R: Rng + ?Sized,
{
    fn select_unit(&mut self, container: &mut Container<'a, R>) -> Option<usize> {
        if container.indices().len() <= 1 {
            return container.indices().first().cloned();
        }

        match self.unit.as_mut() {
            Some(u) => {
                // Sequential order
                while *u < container.population_size() {
                    if container.indices().contains(*u) {
                        return Some(*u);
                    }

                    *u += 1;
                }
            }
            None => {
                // Random order
                return container.indices_draw().cloned();
            }
        }

        unsafe { std::hint::unreachable_unchecked() };
    }
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        if container.indices().is_empty() {
            return;
        }

        self.searcher
            .find_neighbours_of_iter(
                &self.tree,
                container.probabilities(),
                self.tree.data().row_iter(id),
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
                    container.probabilities_mut()[id] += removable_weight * quota;
                    self.decide_unit(container, id);
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
                container.probabilities_mut()[id] += removable_weight * quota;
                self.decide_unit(container, id);
                remaining_weight -= removable_weight;
                sharers -= 1.0;
                i += 1;
            }

            i = j;
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap().map(|r| {
            self.tree.remove_unit(id).unwrap();
            r
        })
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for LocallyCorrelatedPoissonSampling<'a>
where
    R: Rng + ?Sized,
{
    fn select_unit(&mut self, container: &mut Container<'a, R>) -> Option<usize> {
        if container.indices().len() <= 1 {
            return container.indices().first().cloned();
        } else if container.indices().len() == 2 {
            return container.indices_draw().cloned();
        }

        let mut minimum_distance = f64::MAX;
        self.candidates.clear();

        // Loop through all remaining units
        let mut i = 0;
        while i < container.indices().len() {
            let id = *container.indices().get(i).unwrap();
            self.scps
                .searcher
                .find_neighbours_of_id(&self.scps.tree, container.probabilities(), id)
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

        random_element(container.rng(), &self.candidates).cloned()
    }
    #[inline]
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        self.scps
            .update_neighbours(container, id, probability, quota)
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        self.scps.decide_unit(container, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;
    use envisim_utils::matrix::RefMatrix;

    #[test]
    fn cps_sampler() -> Result<(), SamplingError> {
        let mut rng = seeded_rng();
        let mut cps = cps_new(&mut rng, &SampleOptions::new(&PROB_10_E)?)?;
        assert_eq!(cps.decide_selected(7, 0.0), (0.2, -0.8));
        let mut cps = cps_new(&mut rng, &SampleOptions::new(&PROB_10_E)?)?;
        assert_eq!(cps.decide_selected(7, 1.0), (0.2, 0.2));
        Ok(())
    }

    fn decide_and_update<'a, R, T>(
        cps: &mut CorrelatedPoissonSampler<'a, R, T>,
        id: usize,
        rv: f64,
    ) -> (usize, f64, f64)
    where
        R: rand::Rng,
        T: CorrelatedPoissonVariant<'a, R>,
    {
        let (p, q) = cps.decide_selected(id, rv);
        cps.variant.update_neighbours(&mut cps.container, id, p, q);
        (id, p, q)
    }

    #[test]
    fn cps_variant() -> Result<(), SamplingError> {
        let mut rng = seeded_rng();

        let mut cps = cps_new(&mut rng, &SampleOptions::new(&PROB_10_E)?)?;
        decide_and_update(&mut cps, 0, 0.0);
        assert_fvec(&cps.container.probabilities().data()[1..=4], &vec![0.0; 4]);

        let mut cps = cps_new(&mut rng, &SampleOptions::new(&PROB_10_E)?)?;
        decide_and_update(&mut cps, 0, 0.999);
        assert_fvec(&cps.container.probabilities().data()[1..=4], &vec![0.25; 4]);
        Ok(())
    }

    #[test]
    fn scps_variant() -> Result<(), SamplingError> {
        let mut rng = seeded_rng();
        let data = RefMatrix::new(&DATA_10_2, 10);

        let mut cps = scps_new(
            &mut rng,
            SampleOptions::new(&PROB_10_E)?.auxiliaries(&data)?,
        )?;
        decide_and_update(&mut cps, 0, 0.0);
        assert_delta!(cps.container.probabilities()[1], 0.0);
        assert_delta!(cps.container.probabilities()[8], 0.0);
        assert_delta!(cps.container.probabilities()[4], 0.0);
        assert_delta!(cps.container.probabilities()[2], 0.0);

        let mut cps = scps_new(
            &mut rng,
            SampleOptions::new(&PROB_10_E)?.auxiliaries(&data)?,
        )?;
        decide_and_update(&mut cps, 9, 1.0);
        assert_delta!(cps.container.probabilities()[4], 0.25);
        assert_delta!(cps.container.probabilities()[2], 0.25);
        assert_delta!(cps.container.probabilities()[0], 0.25);
        assert_delta!(cps.container.probabilities()[7], 0.25);
        Ok(())
    }

    #[test]
    fn lcps_variant() -> Result<(), SamplingError> {
        let mut rng = seeded_rng();
        let data = RefMatrix::new(&DATA_10_2, 10);

        let mut cps = lcps_new(
            &mut rng,
            SampleOptions::new(&PROB_10_E)?.auxiliaries(&data)?,
        )?;
        assert_eq!(cps.variant.select_unit(&mut cps.container), Some(8));
        decide_and_update(&mut cps, 8, 0.0);
        assert_delta!(cps.container.probabilities()[3], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[5], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[1], 0.0, EPS);

        let mut cps = lcps_new(
            &mut rng,
            SampleOptions::new(&PROB_10_E)?.auxiliaries(&data)?,
        )?;
        decide_and_update(&mut cps, 8, 1.0);
        assert_delta!(cps.container.probabilities()[3], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[5], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[1], 0.25, EPS);
        Ok(())
    }
}
