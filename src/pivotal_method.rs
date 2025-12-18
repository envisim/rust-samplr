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

//! Pivotal method designs

use envisim_utils::indices::Pair;
use envisim_utils::kd_tree::Searcher;
use envisim_utils::pips::{
    Probabilities,
    ProbabilitiesUnequal,
};
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    Enabled,
    ProbabilitySpec,
};
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};
use envisim_utils::utils::usize_to_f64;
use rustc_hash::FxHashSet;

use crate::SamplingError;
use crate::sample_controller::{
    BasicSampleController,
    SampleController,
    SpreadingSampleController,
};

pub trait PivotalMethod<C>
where
    C: SampleController,
{
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    fn sample<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Vec<usize> {
        self.run(rng);
        self.controller_mut().sample_mut().sort_to_vec()
    }
    fn run<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        while self.update_probabilities(rng) {}

        self.controller_mut()
            .unit_decide_last(rng)
            .expect("last unit to be decided");
    }
    fn select_pair<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Pair;
    fn update_probabilities<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> bool {
        let (id1, id2, cont) = match self.select_pair(rng) {
            Pair::More(id1, id2) => (id1, id2, true),
            Pair::Two(id1, id2) => (id1, id2, false),
            _ => {
                return false;
            }
        };

        let one = self.controller().probabilities().one();

        let p1 = self.controller().probabilities().data()[id1];
        let p2 = self.controller().probabilities().data()[id2];
        let psum = p1 + p2;

        if psum == one {
            if self.controller().draw(rng, one) < p1 {
                self.controller_mut()
                    .unit_set_one(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_zero(id2)
                    .expect("id2 to update");
            } else {
                self.controller_mut()
                    .unit_set_zero(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_one(id2)
                    .expect("id2 to update");
            }

            return cont;
        }

        if one < psum {
            if self.controller().draw(rng, one + one - psum) < one - p2 {
                self.controller_mut()
                    .unit_set_one(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_and_decide(id2, psum - one)
                    .expect("id2 to update");
            } else {
                self.controller_mut()
                    .unit_set_and_decide(id1, psum - one)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_one(id2)
                    .expect("id2 to update");
            }
            return cont;
        }

        // psum < one
        if self.controller().probabilities().draw(rng, psum) < p1 {
            self.controller_mut()
                .unit_set_and_decide(id1, psum)
                .expect("id1 to update");
            self.controller_mut()
                .unit_set_zero(id2)
                .expect("id2 to update");
        } else {
            self.controller_mut()
                .unit_set_zero(id1)
                .expect("id1 to update");
            self.controller_mut()
                .unit_set_and_decide(id2, psum)
                .expect("id2 to update");
        }

        cont
    }
}

pub struct SequentialPivotalMethod<P>
where
    P: Probabilities,
{
    controller: BasicSampleController<P>,
    pair: (usize, usize),
}
impl<P> PivotalMethod<BasicSampleController<P>> for SequentialPivotalMethod<P>
where
    P: Probabilities,
{
    fn controller(&self) -> &BasicSampleController<P> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { &mut self.controller }
    fn select_pair<R: RandomNumberGenerator>(&mut self, _: &mut R) -> Pair {
        let pair: Pair = self.controller.indices().into();
        if !pair.is_full() {
            return pair;
        }

        let pop_size = self.controller.population_size();

        // Check if the pair.0 was the unit to disappear
        if !self.controller.indices().contains(self.pair.0) {
            // Check if pair.1 also disappeared
            self.pair.0 = self.pair.1;

            if !self.controller.indices().contains(self.pair.0) {
                self.pair.0 = self
                    .controller
                    .indices()
                    .seq_after(self.pair.0, pop_size)
                    .expect("two units to remain");
            }
        }

        // Now pair.0 is the first remaining unit...set pair.1 to the next remaining unit
        self.pair.1 = self
            .controller
            .indices()
            .seq_after(self.pair.0, pop_size)
            .expect("two units to remain");
        Pair::new(self.pair)
    }
}
impl<P> SequentialPivotalMethod<P>
where
    P: Probabilities,
{
    pub fn new<S, B>(options: &SamplingOptions<'_, P, S, B>) -> Self
    where
        for<'a> BasicSampleController<P>: From<&'a SamplingOptions<'a, P, S, B>>,
    {
        SequentialPivotalMethod {
            controller: options.into(),
            pair: (0, 1),
        }
    }
}
/// Draw a sample using the sequential pivotal method.
/// A variant of the pivotal method where unit competes in order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = spm(&mut rng, p.try_into()?);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
pub fn spm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<P>: From<&'a SamplingOptions<'a, P, S, B>>,
{
    SequentialPivotalMethod::new(options).sample(rng)
}

pub struct RandomPivotalMethod<P>
where
    P: Probabilities,
{
    controller: BasicSampleController<P>,
}
impl<P> PivotalMethod<BasicSampleController<P>> for RandomPivotalMethod<P>
where
    P: Probabilities,
{
    fn controller(&self) -> &BasicSampleController<P> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { &mut self.controller }
    fn select_pair<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Pair {
        let pair: Pair = self.controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let len = self.controller.indices().len();
        let id1 = self.controller.indices().draw(rng).unwrap();
        let k = rng.rusize_to(len - 1);
        let mut id2 = self.controller.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = self.controller.indices().last().unwrap();
        }

        Pair::More(id1, id2)
    }
}
impl<P> RandomPivotalMethod<P>
where
    P: Probabilities,
{
    pub fn new<S, B>(options: &SamplingOptions<'_, P, S, B>) -> Self
    where
        for<'a> BasicSampleController<P>: From<&'a SamplingOptions<'a, P, S, B>>,
    {
        RandomPivotalMethod {
            controller: options.into(),
        }
    }
}
/// Draw a sample using the random pivotal method.
/// A variant of the pivotal method where unit competes in a random order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = rpm(&mut rng, p.try_into()?);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
// pub fn rpm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
// where
//     R: RandomNumberGenerator,
//     P: Probabilities,
// {
//     let controller: BasicSampleController<P> = options.into();
//     RandomPivotalMethod { controller }.sample(rng)
// }
pub fn rpm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<P>: From<&'a SamplingOptions<'a, P, S, B>>,
{
    RandomPivotalMethod::new(options).sample(rng)
}

pub struct LocalPivotalMethod1<'a, P>
where
    P: Probabilities,
{
    controller: SpreadingSampleController<'a, P>,
    candidates: Vec<usize>,
    searcher: Searcher,
}
impl<'a, P> PivotalMethod<SpreadingSampleController<'a, P>> for LocalPivotalMethod1<'a, P>
where
    P: Probabilities,
{
    fn controller(&self) -> &SpreadingSampleController<'a, P> { &self.controller }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, P> { &mut self.controller }
    fn select_pair<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Pair {
        let pair: Pair = self.controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        loop {
            let id1 = self.controller.indices().draw(rng).unwrap();
            self.searcher
                .find_neighbours_of_id(self.controller.tree(), id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;

            while i < self.candidates.len() {
                self.searcher
                    .find_neighbours_of_id(self.controller.tree(), self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().contains(&id1) {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if !self.candidates.is_empty() {
                let id2 = *rng.relement(&self.candidates).unwrap();
                return Pair::More(id1, id2);
            }
        }
    }
}
impl<'a, P> LocalPivotalMethod1<'a, P>
where
    P: Probabilities,
{
    pub fn new<B>(options: &'a SamplingOptions<'a, P, Enabled, B>) -> Self
    where
        SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
    {
        let controller: SpreadingSampleController<'_, P> = options.into();
        let searcher = Searcher::new_1(controller.tree());
        LocalPivotalMethod1 {
            controller,
            candidates: Vec::<usize>::with_capacity(20),
            searcher,
        }
    }
}
/// Draw a sample using the local pivotal method 1.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let opts = SampleOptions::new(&p)?.set_spreading(&m)?.sample(&mut rng, lpm_1)?;
/// let s = lpm_1(&mut rng, opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
pub fn lpm_1<'a, R, P, B>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, P, Enabled, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    LocalPivotalMethod1::new(options).sample(rng)
}

pub struct LocalPivotalMethod1S<'a, P>
where
    P: Probabilities,
{
    controller: SpreadingSampleController<'a, P>,
    candidates: Vec<usize>,
    history: Vec<usize>,
    searcher: Searcher,
}
impl<'a, P> PivotalMethod<SpreadingSampleController<'a, P>> for LocalPivotalMethod1S<'a, P>
where
    P: Probabilities,
{
    fn controller(&self) -> &SpreadingSampleController<'a, P> { &self.controller }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, P> { &mut self.controller }
    fn select_pair<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Pair {
        let pair: Pair = self.controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        while let Some(&id) = self.history.last() {
            if self.controller.indices().contains(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.is_empty() {
            self.history
                .push(self.controller.indices().draw(rng).unwrap());
        }

        loop {
            let id1 = *self.history.last().unwrap();
            self.searcher
                .find_neighbours_of_id(self.controller.tree(), id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;
            let mut len = self.candidates.len();

            while i < len {
                self.searcher
                    .find_neighbours_of_id(self.controller.tree(), self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().contains(&id1) {
                    i += 1;
                } else {
                    // If we does not find any compatible matches, we use the candidates to continue our seach
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2 = *rng.relement(&self.candidates[0..len]).unwrap();
                return Pair::More(id1, id2);
            }

            if self.history.len() == self.controller.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history.push(*rng.relement(&self.candidates).unwrap());
        }
    }
}
impl<'a, P> LocalPivotalMethod1S<'a, P>
where
    P: Probabilities,
{
    pub fn new<B>(options: &'a SamplingOptions<'a, P, Enabled, B>) -> Self
    where
        SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
    {
        let controller: SpreadingSampleController<'_, P> = options.into();
        let searcher = Searcher::new_1(controller.tree());
        let remaining_units = controller.indices().len();
        LocalPivotalMethod1S {
            controller,
            candidates: Vec::<usize>::with_capacity(20),
            history: Vec::<usize>::with_capacity(remaining_units),
            searcher,
        }
    }
}
/// Draw a sample using the local pivotal method 1.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let opts = SampleOptions::new(&p)?.set_spreading(&m)?.sample(&mut rng, lpm_1)?;
/// let s = lpm_1s(&mut rng, opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
pub fn lpm_1s<'a, R, P, B>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, P, Enabled, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    LocalPivotalMethod1S::new(options).sample(rng)
}

pub struct LocalPivotalMethod2<'a, P>
where
    P: Probabilities,
{
    controller: SpreadingSampleController<'a, P>,
    searcher: Searcher,
}
impl<'a, P> PivotalMethod<SpreadingSampleController<'a, P>> for LocalPivotalMethod2<'a, P>
where
    P: Probabilities,
{
    fn controller(&self) -> &SpreadingSampleController<'a, P> { &self.controller }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, P> { &mut self.controller }
    fn select_pair<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Pair {
        let pair: Pair = self.controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let id1 = self.controller.indices().draw(rng).unwrap();
        self.searcher
            .find_neighbours_of_id(self.controller.tree(), id1)
            .unwrap();
        let id2 = *rng.relement(self.searcher.neighbours()).unwrap();

        Pair::More(id1, id2)
    }
}
impl<'a, P> LocalPivotalMethod2<'a, P>
where
    P: Probabilities,
{
    pub fn new<B>(options: &'a SamplingOptions<'a, P, Enabled, B>) -> Self
    where
        SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
    {
        let controller: SpreadingSampleController<'_, P> = options.into();
        let searcher = Searcher::new_1(controller.tree());
        LocalPivotalMethod2 {
            controller,
            searcher,
        }
    }
}
/// Draw a sample using the local pivotal method 2.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let opts = SampleOptions::new(&p)?.set_spreading(&m)?.sample(&mut rng, lpm_1)?;
/// let s = lpm_2(&mut rng, opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
pub fn lpm_2<'a, R, P, B>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, P, Enabled, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    SpreadingSampleController<'a, P>: From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    LocalPivotalMethod2::new(options).sample(rng)
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
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SampleOptions::new(&p)?.set_spreading(&m)?;
/// let sizes = [3, 2];
/// let s = hierarchical_lpm_2(&mut rng, &options, &sizes)?;
///
/// assert_eq!(s.len(), 2);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
#[inline]
pub fn hierarchical_lpm_2<'a, R, P, B>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, P, Enabled, B>,
    sizes: &[usize],
) -> Result<Vec<Vec<usize>>, SamplingError>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    // Check validity of probabilities and sizes
    let sizes_sum = sizes.iter().sum();
    match options.probabilities().spec() {
        ProbabilitySpec::Equal { sample_size } => {
            if *sample_size != sizes_sum {
                return Err(SamplingError::IncorrectStratification);
            }
        }
        _ => {
            let psum = options.probabilities().slice().iter().sum::<f64>();
            if (psum.round() - psum).abs() > options.eps() {
                return Err(SamplingError::IncorrectStratification);
            }
            let sizes_sum = usize_to_f64(sizes_sum);
            if psum.round() != sizes_sum {
                return Err(SamplingError::IncorrectStratification);
            }
        }
    };

    // if sizes.len() == 1 {
    //     return Ok(vec![lpm_2(rng, options)]);
    // }

    let mut pm = {
        let controller: SpreadingSampleController<'_, ProbabilitiesUnequal> = options.into();
        let searcher = Searcher::new_1(controller.tree());
        LocalPivotalMethod2 {
            controller,
            searcher,
        }
    };
    pm.run(rng);

    if sizes.len() == 1 {
        return Ok(vec![pm.controller.sample_mut().sort_to_vec()]);
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());

    let mut main_sample: FxHashSet<usize> =
        FxHashSet::from_iter(pm.controller.sample().get().iter().cloned());

    for &size in sizes[0..sizes.len() - 1].iter() {
        assert!(pm.controller.indices().is_empty());

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.controller.sample_mut().clear();

        let prob = usize_to_f64(size) / usize_to_f64(main_sample.len());

        // Reset probs and add to indices/tree
        for id in 0..pm.controller.population_size() {
            if main_sample.contains(&id) {
                pm.controller.probabilities_mut().set(id, prob).unwrap();
                pm.controller.indices_mut().insert(id).unwrap();
                pm.controller.tree_mut().insert_unit(id).unwrap();
            } else {
                pm.controller.probabilities_mut().set_zero(id);
            }
        }

        pm.run(rng);

        let s = pm.controller.sample_mut().sort_to_vec();
        s.iter().for_each(|id| {
            main_sample.remove(id);
        });

        return_sample.push(s);
    }

    let mut s: Vec<usize> = main_sample.into_iter().collect();
    s.sort_unstable();
    return_sample.push(s);

    Ok(return_sample)
}
