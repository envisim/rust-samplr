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

//! Spatial pivotal methods

use envisim_utils::indices::Pair;
use envisim_utils::kd_tree::searcher::{
    NearestNeighbourSearcher,
    NeighbourSlice,
};
use envisim_utils::kd_tree::{
    PointSet,
    Tree,
};
use envisim_utils::probabilities::{
    Probability,
    RealProbabilityValue,
};
use envisim_utils::random::{
    Rand,
    Rng,
    random_element,
};
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    ProbabilityOptions,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use num_traits::ToPrimitive;
use rustc_hash::FxHashSet;

use super::runner::{
    PivotalRunner,
    PivotalStrategy,
};
use crate::error::{
    SamplingError,
    SamplingResult,
};

/// Returns true if `id_n` has `id_org` as a nearest neighbour. If `id_n` is amongst the nearest
/// neighbours of `id_org`, they are mutual nns
#[inline]
fn is_mutual_nn<P>(
    searcher: &mut NearestNeighbourSearcher<P>,
    tree: &Tree<'_, P>,
    id_org: P::Id,
    id_n: P::Id,
) -> bool
where
    P: PointSet,
{
    searcher
        .reset_from_unit(tree.data(), id_n)
        .expect("id_n to exist")
        .search(tree)
        .expect("nn to be found");
    searcher.neighbours().contains_id(id_org)
}

#[must_use]
pub struct LocalStrategy1<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<usize>,
}
impl<P> LocalStrategy1<P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new [`PivotalRunner`]using the LPM1 strategy
    #[inline]
    pub fn new<PO, BAL>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    ) -> PivotalRunner<Self, PO::Native, Tree<'_, P>>
    where
        PO: ProbabilityOptions,
    {
        let controller = options.to_spreading_controller();
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<usize>::with_capacity(20);
        PivotalRunner {
            controller,
            strategy: Self {
                searcher,
                candidates,
            },
        }
    }
}
impl<PROB, P> PivotalStrategy<PROB, Tree<'_, P>> for LocalStrategy1<P>
where
    P: PointSet<Id = usize>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        controller: &mut SampleController<PROB, Tree<'_, P>>,
        rng: &mut R,
    ) -> Pair
    where
        R: Rand<usize>,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        loop {
            let id1 = controller
                .indices()
                .draw(rng)
                .expect("indices to have units");
            self.searcher
                .reset_from_unit(controller.tree().data(), id1)
                .expect("id1 to exist")
                .search(controller.tree())
                .expect("nn to be found");
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend(self.searcher.neighbours().to_neighbour_id_iter());

            {
                let mut i = 0_usize;
                while i < self.candidates.len() {
                    if is_mutual_nn(
                        &mut self.searcher,
                        controller.tree(),
                        id1,
                        self.candidates[i],
                    ) {
                        i += 1;
                    } else {
                        self.candidates.swap_remove(i);
                    }
                }
            }

            if !self.candidates.is_empty() {
                let id2 =
                    *random_element(rng, &self.candidates).expect("candidates to have elements");
                return Pair::More(id1, id2);
            }
        }
    }
}

#[must_use]
pub struct LocalStrategy1S<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<usize>,
    /// History of potential minimal nns
    history: Vec<usize>,
}
impl<P> LocalStrategy1S<P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new [`PivotalRunner`]using the LPM1 fast strategy
    #[inline]
    pub fn new<PO, BAL>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    ) -> PivotalRunner<Self, PO::Native, Tree<'_, P>>
    where
        PO: ProbabilityOptions,
    {
        let controller = options.to_spreading_controller();
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<usize>::with_capacity(20);
        let history = Vec::<usize>::with_capacity(controller.indices().len());
        PivotalRunner {
            controller,
            strategy: Self {
                searcher,
                candidates,
                history,
            },
        }
    }
}
impl<PROB, P> PivotalStrategy<PROB, Tree<'_, P>> for LocalStrategy1S<P>
where
    P: PointSet<Id = usize>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        controller: &mut SampleController<PROB, Tree<'_, P>>,
        rng: &mut R,
    ) -> Pair
    where
        R: Rand<usize>,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        while let Some(&id) = self.history.last() {
            if controller.indices().contains(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.is_empty() {
            self.history.push(
                controller
                    .indices()
                    .draw(rng)
                    .expect("indices to have units"),
            );
        }

        loop {
            let id1 = *self.history.last().expect("history to not be empty");

            self.searcher
                .reset_from_unit(controller.tree().data(), id1)
                .expect("id1 to exist")
                .search(controller.tree())
                .expect("nn to be found");
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any of the potential
            // equals is a match
            self.candidates
                .extend(self.searcher.neighbours().to_neighbour_id_iter());

            // Partition candidates into compatible and non-compatible matches
            let mut left = 0;
            let mut right = self.candidates.len();
            while left < right {
                if is_mutual_nn(
                    &mut self.searcher,
                    controller.tree(),
                    id1,
                    self.candidates[left],
                ) {
                    left += 1;
                } else {
                    right -= 1;
                    self.candidates.swap(left, right);
                }
            }

            // Some mutual nn has been found
            if left > 0 {
                let id2 = *random_element(rng, &self.candidates[0..left]).expect("left > 0");
                return Pair::More(id1, id2);
            }
            // If no mutual nn has been found, we select one of the candidates by random to be the
            // next search unit ... but first we clear the history if it has become to long
            if self.history.len() == controller.population_size().get() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history.push(
                *random_element(rng, &self.candidates).expect("candidates to have positive length"),
            );
        }
    }
}

#[must_use]
pub struct LocalStrategy2<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
}
impl<P> LocalStrategy2<P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new [`PivotalRunner`]using the LPM2 strategy
    #[inline]
    pub fn new<PO, BAL>(
        options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    ) -> PivotalRunner<Self, PO::Native, Tree<'_, P>>
    where
        PO: ProbabilityOptions,
        P: PointSet,
    {
        let controller = options.to_spreading_controller();
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        PivotalRunner {
            controller,
            strategy: Self { searcher },
        }
    }
}
impl<PROB, P> PivotalStrategy<PROB, Tree<'_, P>> for LocalStrategy2<P>
where
    P: PointSet<Id = usize>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        controller: &mut SampleController<PROB, Tree<'_, P>>,
        rng: &mut R,
    ) -> Pair
    where
        R: Rand<usize>,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let id1 = controller
            .indices()
            .draw(rng)
            .expect("indices to have units");
        self.searcher
            .reset_from_unit(controller.tree().data(), id1)
            .expect("id1 to exist")
            .search(controller.tree())
            .expect("nn to be found");
        let id2 = random_element(rng, self.searcher.neighbours())
            .expect("neighbours to have positive length")
            .id();

        Pair::More(id1, id2)
    }
}

pub trait LocalPivotalSampling<R>
where
    R: Rng,
{
    /// Draw a sample using the local pivotal method 1.
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
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_1(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_1(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using the local pivotal method 1.
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
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_1s(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_1s(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using the local pivotal method 2.
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
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_2(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_2(&self, rng: &mut R) -> Vec<usize>;
}
impl<R, PO, P, BAL> LocalPivotalSampling<R> for SamplingOptions<PO, SpreadingOptions<P>, BAL>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions,
    P: PointSet<Id = usize>,
{
    #[inline]
    fn lpm_1(&self, rng: &mut R) -> Vec<usize> { LocalStrategy1::new(self).sample(rng) }
    #[inline]
    fn lpm_1s(&self, rng: &mut R) -> Vec<usize> { LocalStrategy1S::new(self).sample(rng) }
    #[inline]
    fn lpm_2(&self, rng: &mut R) -> Vec<usize> { LocalStrategy2::new(self).sample(rng) }
}

/// Draw a sample using the hierarchical local pivotal method 2.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
/// Selects an initial sample using [`lpm_2`], and splits this sample into subsamples of given
/// `sizes`, using successive, hierarchical selection with `lpm_2`.
/// `sizes` must sum to the sum of `probabilities`.
///
/// # Examples
/// ```
/// # use envisim_samplr::*;
/// # use envisim_samplr::pivotal_method::hierarchical_lpm_2;
/// # use envisim_utils::random::*;
/// # use envisim_utils::matrix::Matrix;
/// let mut rng = try_sys_rng().unwrap();
/// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(p.into())?.set_spreading(m)?;
/// let sizes = [3, 2];
/// let s = hierarchical_lpm_2(&mut rng, &options, &sizes)?;
/// assert_eq!(s.len(), 2);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
///
/// # Errors
/// Returns error if
///
/// - `sizes` does not sum to `sample_size`
/// - probabilities does not sum to an integer
#[expect(
    clippy::missing_panics_doc,
    clippy::panic_in_result_fn,
    reason = "panic implies bug"
)]
#[inline]
pub fn hierarchical_lpm_2<R, PO, P, BAL>(
    rng: &mut R,
    options: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    sizes: &[usize],
) -> SamplingResult<Vec<Vec<usize>>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize>,
{
    // Check validity of probabilities and sizes
    let sizes_sum = sizes.iter().sum();
    let sample_size = options.probabilities().sample_size();
    let psum = options.probabilities().sample_size_real();
    if sample_size != sizes_sum || !options.eps().difference_is_zero(psum.round(), psum) {
        return Err(SamplingError::IncorrectStratification);
    }

    // Cannot use ::new, controller needs to be float
    let mut pm = {
        let controller = options.to_spreading_controller_real();
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        PivotalRunner {
            controller,
            strategy: LocalStrategy2 { searcher },
        }
    };
    pm.run(rng);

    if sizes.len() == 1 {
        return Ok(vec![pm.controller.to_sorted_sample_vec()]);
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());
    let mut main_sample: FxHashSet<usize> = pm.controller.sample().get().iter().copied().collect();

    for &size in &sizes[0..sizes.len() - 1] {
        assert!(pm.controller.indices().is_empty(), "indices is empty");

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.controller.sample_mut().clear();

        let prob = Probability::new_real(
            size.to_f64().expect("size to convert to f64")
                / main_sample
                    .len()
                    .to_f64()
                    .expect("main sample length convert to f64"),
            options.eps(),
        )
        .expect("fraction to be in [0,1]");

        // Reset probs and add to indices/tree
        for id in 0..pm.controller.population_size().get() {
            if main_sample.contains(&id) {
                pm.controller.probabilities_mut().set(id, prob);
                pm.controller.indices_mut().insert(id);
                pm.controller
                    .tree_mut()
                    .insert_unit(id)
                    .expect("id not to already have been inserted into tree");
            } else {
                pm.controller.probabilities_mut().set_zero(id);
            }
        }

        pm.run(rng);

        let s = {
            pm.controller.sample_mut().sort();
            pm.controller.sample().get().to_vec()
        };
        for id in &s {
            main_sample.remove(id);
        }

        return_sample.push(s);
    }

    // Take whatever's left and make into a sample
    let mut s: Vec<usize> = main_sample.into_iter().collect();
    s.sort_unstable();
    return_sample.push(s);

    Ok(return_sample)
}
