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

//! Pivotal method designs

use crate::utils::Container;
pub use crate::SampleOptions;
use envisim_utils::kd_tree::{Node, Searcher};
use envisim_utils::utils::{random_element, sum, usize_to_f64};
use envisim_utils::{InputError, SamplingError};
use rand::Rng;
use rustc_hash::FxHashSet;

type Pair = (usize, usize);

pub trait PivotalMethodVariant<'a, R>
where
    R: Rng + ?Sized,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)>;
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct PivotalMethodSampler<'a, R, T>
where
    R: Rng + ?Sized,
    T: PivotalMethodVariant<'a, R>,
{
    container: Box<Container<'a, R>>,
    variant: Box<T>,
}

pub struct SequentialPivotalMethod {
    pair: Pair,
}
pub struct RandomPivotalMethod {}
pub struct LocalPivotalMethod1<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
    candidates: Vec<usize>,
}
pub struct LocalPivotalMethod1S<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
    candidates: Vec<usize>,
    history: Vec<usize>,
}
pub struct LocalPivotalMethod2<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
}

/// Draw a sample using the sequential pivotal method.
/// A variant of the pivotal method where unit competes in order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = SampleOptions::new(&p)?.sample(&mut rng, spm)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
#[inline]
pub fn spm<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    spm_new(rng, options)?.sample_with_return()
}
#[inline]
fn spm_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions,
) -> Result<PivotalMethodSampler<'a, R, SequentialPivotalMethod>, SamplingError>
where
    R: Rng + ?Sized,
{
    Ok(PivotalMethodSampler {
        container: Container::new_boxed(rng, options)?,
        variant: Box::new(SequentialPivotalMethod { pair: (0, 1) }),
    })
}

/// Draw a sample using the random pivotal method.
/// A variant of the pivotal method where unit competes in a random order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = SampleOptions::new(&p)?.sample(&mut rng, rpm)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
#[inline]
pub fn rpm<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    rpm_new(rng, options)?.sample_with_return()
}
#[inline]
fn rpm_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions,
) -> Result<PivotalMethodSampler<'a, R, RandomPivotalMethod>, SamplingError>
where
    R: Rng + ?Sized,
{
    Ok(PivotalMethodSampler {
        container: Container::new_boxed(rng, options)?,
        variant: Box::new(RandomPivotalMethod {}),
    })
}

/// Draw a sample using the local pivotal method 1.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::Matrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.sample(&mut rng, lpm_1)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
#[inline]
pub fn lpm_1<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    lpm_1_new(rng, options)?.sample_with_return()
}
#[inline]
fn lpm_1_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<PivotalMethodSampler<'a, R, LocalPivotalMethod1<'a>>, SamplingError>
where
    R: Rng + ?Sized,
{
    options.check_spatially_balanced()?;
    let container = Container::new_boxed(rng, options)?;
    let tree = options.build_node(&mut container.indices().to_vec())?;
    let searcher = Box::new(Searcher::new(&tree, 1)?);

    Ok(PivotalMethodSampler {
        container,
        variant: Box::new(LocalPivotalMethod1 {
            tree,
            searcher,
            candidates: Vec::<usize>::with_capacity(20),
        }),
    })
}

/// Draw a sample using the local pivotal method 1S.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::Matrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.sample(&mut rng, lpm_1s)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Prentius, W. (2024). Manuscript.
#[inline]
pub fn lpm_1s<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    lpm_1s_new(rng, options)?.sample_with_return()
}
#[inline]
fn lpm_1s_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<PivotalMethodSampler<'a, R, LocalPivotalMethod1S<'a>>, SamplingError>
where
    R: Rng + ?Sized,
{
    options.check_spatially_balanced()?;
    let container = Container::new_boxed(rng, options)?;
    let tree = options.build_node(&mut container.indices().to_vec())?;
    let searcher = Box::new(Searcher::new(&tree, 1)?);
    let remaining_units = container.indices().len();

    Ok(PivotalMethodSampler {
        container,
        variant: Box::new(LocalPivotalMethod1S {
            tree,
            searcher,
            candidates: Vec::<usize>::with_capacity(20),
            history: Vec::<usize>::with_capacity(remaining_units),
        }),
    })
}

/// Draw a sample using the local pivotal method 2.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::Matrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let s = SampleOptions::new(&p)?.auxiliaries(&m)?.sample(&mut rng, lpm_1)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
#[inline]
pub fn lpm_2<R>(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    lpm_2_new(rng, options)?.sample_with_return()
}
#[inline]
fn lpm_2_new<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<PivotalMethodSampler<'a, R, LocalPivotalMethod2<'a>>, SamplingError>
where
    R: Rng + ?Sized,
{
    options.check_spatially_balanced()?;
    let container = Container::new_boxed(rng, options)?;
    let tree = options.build_node(&mut container.indices().to_vec())?;
    let searcher = Box::new(Searcher::new(&tree, 1)?);

    Ok(PivotalMethodSampler {
        container,
        variant: Box::new(LocalPivotalMethod2 { tree, searcher }),
    })
}

impl<'a, R, T> PivotalMethodSampler<'a, R, T>
where
    R: Rng + ?Sized,
    T: PivotalMethodVariant<'a, R>,
{
    #[inline]
    fn sample_with_return(&mut self) -> Result<Vec<usize>, SamplingError> {
        Ok(self.sample().get_sorted_sample().to_vec())
    }
    #[inline]
    fn sample(&mut self) -> &mut Self {
        while let Some(units) = self.variant.select_units(&mut self.container) {
            let rv = self.container.rng().gen::<f64>();
            self.update_probabilities(units, rv);
        }

        if let Some(id) = self.container.update_last_unit() {
            self.variant.decide_unit(&mut self.container, id);
        }

        self
    }
    #[inline]
    fn update_probabilities(&mut self, (id1, id2): Pair, rv: f64) {
        let mut p1 = self.container.probabilities()[id1];
        let mut p2 = self.container.probabilities()[id2];
        let psum = p1 + p2;

        if psum > 1.0 {
            if 1.0 - p2 > rv * (2.0 - psum) {
                p1 = 1.0;
                p2 = psum - 1.0;
            } else {
                p1 = psum - 1.0;
                p2 = 1.0;
            }
        } else if p2 > rv * psum {
            // psum <= 1.0
            p1 = 0.0;
            p2 = psum;
        } else {
            // psum <= 1.0
            p1 = psum;
            p2 = 0.0;
        }

        self.container.probabilities_mut()[id1] = p1;
        self.container.probabilities_mut()[id2] = p2;
        self.variant.decide_unit(&mut self.container, id1);
        self.variant.decide_unit(&mut self.container, id2);
    }
    #[inline]
    fn get_sample(&mut self) -> &[usize] {
        self.container.sample().get()
    }
    #[inline]
    fn get_sorted_sample(&mut self) -> &[usize] {
        self.container.sample_mut().sort().get()
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for SequentialPivotalMethod
where
    R: Rng + ?Sized,
{
    #[inline]
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        if container.indices().len() <= 1 {
            return None;
        }

        if !container.indices().contains(self.pair.0) {
            self.pair.0 = self.pair.1;

            while !container.indices().contains(self.pair.0) {
                self.pair.0 += 1;

                if self.pair.0 >= container.population_size() {
                    panic!("spm looped past last unit");
                }
            }

            self.pair.1 = self.pair.0 + 1;
        }

        while !container.indices().contains(self.pair.1) {
            self.pair.1 += 1;

            if self.pair.1 >= container.population_size() {
                panic!("spm looped past last unit");
            }
        }

        Some(self.pair)
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap()
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for RandomPivotalMethod
where
    R: Rng + ?Sized,
{
    #[inline]
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_draw().unwrap();
        let k = container.rng().gen_range(0..(len - 1));
        let mut id2 = *container.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = *container.indices().last().unwrap();
        }

        Some((id1, id2))
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap()
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1<'a>
where
    R: Rng + ?Sized,
{
    #[inline]
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        loop {
            let id1 = *container.indices_draw().unwrap();
            self.searcher
                .find_neighbours_of_id(&self.tree, id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;

            while i < self.candidates.len() {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if !self.candidates.is_empty() {
                let id2 = *random_element(container.rng(), &self.candidates).unwrap();
                return Some((id1, id2));
            }
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

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1S<'a>
where
    R: Rng + ?Sized,
{
    #[inline]
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        while let Some(&id) = self.history.last() {
            if container.indices().contains(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.is_empty() {
            self.history.push(*container.indices_draw().unwrap());
        }

        loop {
            let id1 = *self.history.last().unwrap();
            self.searcher
                .find_neighbours_of_id(&self.tree, id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;
            let mut len = self.candidates.len();

            while i < len {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    // If we does not find any compatible matches, we use the candidates to continue our seach
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2 = *random_element(container.rng(), &self.candidates[0..len]).unwrap();
                return Some((id1, id2));
            }

            if self.history.len() == container.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history
                .push(*random_element(container.rng(), &self.candidates).unwrap());
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

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod2<'a>
where
    R: Rng + ?Sized,
{
    #[inline]
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_draw().unwrap();
        self.searcher
            .find_neighbours_of_id(&self.tree, id1)
            .unwrap();
        let id2 = *random_element(container.rng(), self.searcher.neighbours()).unwrap();

        Some((id1, id2))
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap().map(|r| {
            self.tree.remove_unit(id).unwrap();
            r
        })
    }
}

/// Draw a sample using the hierarchical local pivotal method 2.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
/// Selects an initial sample using [`lpm_2`], and splits this sample into subsamples of given
/// `sizes`, using successive, hierarchical selection with `lpm_2`.
/// `sizes` must sum to the sum of `probabilities`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::Matrix;
/// use rand::{rngs::SmallRng, SeedableRng};
///
/// let mut rng = SmallRng::from_entropy();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let mut options = SampleOptions::new(&p)?;
/// options.auxiliaries(&m)?;
/// let sizes = [3, 2];
/// let s = hierarchical_lpm_2(&mut rng, &options, &sizes)?;
///
/// assert_eq!(s.len(), 2);
/// # Ok::<(), envisim_utils::SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
#[inline]
pub fn hierarchical_lpm_2<R>(
    rng: &mut R,
    options: &SampleOptions,
    sizes: &[usize],
) -> Result<Vec<Vec<usize>>, SamplingError>
where
    R: Rng + ?Sized,
{
    InputError::check_integer_approx_equal(
        sum(options.probabilities),
        usize_to_f64(sizes.iter().sum()),
        options.eps,
    )?;
    InputError::check_empty(sizes)?;

    if sizes.len() == 1 {
        return Ok(vec![lpm_2(rng, options)?]);
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());
    let mut pm = lpm_2_new(rng, options)?;

    let mut main_sample: FxHashSet<usize> = pm.sample().get_sample().iter().cloned().collect();

    for (i, &size) in sizes[0..sizes.len() - 1].iter().enumerate() {
        assert!(pm.container.indices().is_empty());

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.container.sample_mut().clear();

        let prob = usize_to_f64(size) / usize_to_f64(main_sample.len());

        // Reset probs and add to indices/tree
        for id in 0..pm.container.population_size() {
            if main_sample.contains(&id) {
                pm.container.probabilities_mut()[id] = prob;
                pm.container.indices_mut().insert(id).unwrap();
                pm.variant.tree.insert_unit(id).unwrap();
            } else {
                pm.container.probabilities_mut()[id] = 0.0;
            }
        }

        pm.sample();
        return_sample.push(Vec::<usize>::with_capacity(pm.get_sample().len()));

        for &id in pm.get_sorted_sample().iter() {
            return_sample[i].push(id);
            main_sample.remove(&id);
        }
    }

    return_sample.push(main_sample.into_iter().collect());

    for s in return_sample.iter_mut() {
        s.sort_unstable();
    }

    Ok(return_sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;

    #[test]
    fn update_probabilities() -> Result<(), SamplingError> {
        let mut rng = seeded_rng();
        let mut pm = spm_new(&mut rng, &SampleOptions::new(&PROB_10_E)?)?;
        pm.update_probabilities((0, 1), 0.0);
        assert_delta!(pm.container.probabilities()[0], 0.0);
        assert_delta!(pm.container.probabilities()[1], 0.4);
        assert!(!pm.container.indices().contains(0));
        assert!(pm.container.indices().contains(1));
        pm.update_probabilities((2, 1), 0.0);
        pm.update_probabilities((3, 1), 0.0);
        pm.update_probabilities((4, 1), 0.0);
        assert!(!pm.container.indices().contains(1));
        assert!(pm.container.sample().get().contains(&1));
        Ok(())
    }
}
