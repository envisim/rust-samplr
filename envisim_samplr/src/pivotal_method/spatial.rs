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

//! Spatial pivotal methods.

use envisim_utils::indices::Pair;
use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::{
    NearestNeighbourSearcher,
    Neighbour,
    NeighbourView,
};
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::{
    Rand,
    Rng,
    random_element,
};
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    EqualProbabilities,
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::PointSet;
use num_traits::FromPrimitive;
use rustc_hash::FxHashSet;

use super::runner::{
    PivotalStrategy,
    pivotal_runner,
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
    searcher
        .neighbours()
        .iter()
        .map(Neighbour::id)
        .any(|id| *id == id_org)
}

/// The local pivotal method variant 1
#[must_use]
struct LocalStrategy1<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<P::Id>,
}
impl<PR, P> PivotalStrategy<PR, Tree<'_, P>> for LocalStrategy1<P>
where
    PR: ProbabilityStore,
    P: PointSet<Id = PR::Id>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        rng: &mut R,
        controller: &SampleController<PR, Tree<'_, P>>,
    ) -> Pair<PR::Id>
    where
        R: Rand<usize>,
    {
        let pair: Pair<PR::Id> = controller.indices().into();
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
                .extend(self.searcher.neighbours().iter().map(Neighbour::id));

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

/// The local pivotal method variant 1S
#[must_use]
pub struct LocalStrategy1S<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
    /// The candidates to be selected as deciding unit
    candidates: Vec<P::Id>,
    /// History of potential minimal nns
    history: Vec<P::Id>,
}
impl<PR, P> PivotalStrategy<PR, Tree<'_, P>> for LocalStrategy1S<P>
where
    PR: ProbabilityStore,
    P: PointSet<Id = PR::Id>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        rng: &mut R,
        controller: &SampleController<PR, Tree<'_, P>>,
    ) -> Pair<PR::Id>
    where
        R: Rand<usize>,
    {
        let pair: Pair<PR::Id> = controller.indices().into();
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
                .extend(self.searcher.neighbours().iter().map(Neighbour::id));

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

/// The local pivotal method variant 2
#[must_use]
pub struct LocalStrategy2<P>
where
    P: PointSet,
{
    /// The searcher to be used to find the neighbours of the selected unit
    searcher: NearestNeighbourSearcher<P>,
}
impl<PR, P> PivotalStrategy<PR, Tree<'_, P>> for LocalStrategy2<P>
where
    PR: ProbabilityStore,
    P: PointSet<Id = PR::Id>,
{
    #[inline]
    fn select_pair<R>(
        &mut self,
        rng: &mut R,
        controller: &SampleController<PR, Tree<'_, P>>,
    ) -> Pair<PR::Id>
    where
        R: Rand<usize>,
    {
        let pair: Pair<PR::Id> = controller.indices().into();
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

        Pair::More(id1, *id2)
    }
}

/// Provides local pivotal sampling methods
pub trait LocalPivotalSampling<ID, R>
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
    /// let s = SamplingOptions::new(p)?
    ///     .set_spreading(m)?
    ///     .lpm_1(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_1(&self, rng: &mut R) -> Vec<ID>;
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
    /// let s = SamplingOptions::new(p)?
    ///     .set_spreading(m)?
    ///     .lpm_1s(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_1s(&self, rng: &mut R) -> Vec<ID>;
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
    /// let s = SamplingOptions::new(p)?
    ///     .set_spreading(m)?
    ///     .lpm_2(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn lpm_2(&self, rng: &mut R) -> Vec<ID>;
}
impl<R, PO, P> LocalPivotalSampling<PO::Id, R> for SamplingOptions<PO, SpreadingOptions<P>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec,
    P: PointSet<Id = PO::Id>,
{
    #[inline]
    fn lpm_1(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_spreading(self);
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<P::Id>::with_capacity(20);
        let strategy = LocalStrategy1 {
            searcher,
            candidates,
        };
        pivotal_runner(rng, controller, strategy).to_sorted_sample_vec()
    }
    #[inline]
    fn lpm_1s(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_spreading(self);
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<PO::Id>::with_capacity(20);
        let history = Vec::<PO::Id>::with_capacity(controller.indices().len());
        let strategy = LocalStrategy1S {
            searcher,
            candidates,
            history,
        };
        pivotal_runner(rng, controller, strategy).to_sorted_sample_vec()
    }
    #[inline]
    fn lpm_2(&self, rng: &mut R) -> Vec<PO::Id> {
        let controller = SampleController::new_spreading(self);
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let strategy = LocalStrategy2 { searcher };
        pivotal_runner(rng, controller, strategy).to_sorted_sample_vec()
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
/// # use envisim_samplr::*;
/// # use envisim_samplr::pivotal_method::hierarchical_lpm_2;
/// # use envisim_utils::random::*;
/// # use envisim_utils::matrix::Matrix;
/// let mut rng = try_sys_rng().unwrap();
/// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(p)?.set_spreading(m)?;
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
#[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
#[inline]
pub fn hierarchical_lpm_2<R, PO, P>(
    rng: &mut R,
    options: &SamplingOptions<PO, SpreadingOptions<P>>,
    sizes: &[usize],
) -> SamplingResult<Vec<Vec<PO::Id>>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Id: Copy>,
    P: PointSet<Id = PO::Id>,
{
    // Early return for degenerate case
    if sizes.is_empty() {
        return Ok(vec![]);
    }

    // Check validity of probabilities and sizes
    let sizes_sum: usize = sizes.iter().sum();
    let sample_size: usize = options.probabilities().sample_size();
    if sizes_sum != sample_size
        || options.probabilities().sample_size_real()
            != <PO::Real as FromPrimitive>::from_usize(sample_size)
                .expect("sample size conv to PO::Real")
    {
        return Err(SamplingError::IncorrectStratification);
    }

    // Early return for uni case
    if sizes.len() == 1 {
        let sample = options.lpm_2(rng);
        return Ok(vec![sample]);
    }

    let mut main_sample: FxHashSet<PO::Id> = options.lpm_2(rng).into_iter().collect();
    let mut return_sample = Vec::<Vec<PO::Id>>::with_capacity(sizes.len());

    for &size in sizes.iter().take(sizes.len() - 1) {
        if size == 0 {
            return_sample.push(vec![]);
            continue;
        }

        let probs =
            EqualProbabilities::with_ids(&main_sample, size).expect("size < main_sample.len");
        let s_opts = SamplingOptions::with_spec(probs)
            .set_spreading(options.spreading())
            .expect("ids to exist in spreading");
        let sample = s_opts.lpm_2(rng);

        for id in &sample {
            main_sample.remove(id);
        }

        return_sample.push(sample);
    }

    // remaining units is its own sample
    let mut sample: Vec<PO::Id> = main_sample.into_iter().collect();
    sample.sort_unstable();
    return_sample.push(sample);

    Ok(return_sample)
}
