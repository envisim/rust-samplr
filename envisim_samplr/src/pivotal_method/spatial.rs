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

use envisim_utils::indices::Pair;
use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::{
    NearestNeighbourSearcher,
    NeighbourSlice,
};
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::{
    SampleController,
    SpreadingSampleController,
};
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
};
use envisim_utils::spatial::{
    Number,
    PointSet,
};

use super::runner::{
    PivotalRunner,
    PivotalStrategy,
};
use crate::error::SamplingResult;

fn is_mutual_nn<P, N>(
    searcher: &mut NearestNeighbourSearcher<N>,
    tree: &Tree<'_, N, P>,
    id_org: usize,
    id_n: usize,
) -> bool
where
    P: PointSet<N>,
    N: Number,
{
    searcher
        .reset_from_unit(tree.data(), id_n)
        .expect("id_n to exist")
        .search(tree)
        .expect("nn to be found");
    searcher.neighbours().contains_id(id_org)
}

pub type PivotalRunnerLocal1<'b, PT, P, N> =
    PivotalRunner<SpreadingSampleController<'b, PT, N, P>, LocalStrategy1<N>>;
pub struct LocalStrategy1<N> {
    searcher: NearestNeighbourSearcher<N>,
    candidates: Vec<usize>,
}
impl<N> LocalStrategy1<N> {
    pub fn new<PS, SOP, BOP>(
        options: &SamplingOptions<PS, SOP, BOP>,
    ) -> SamplingResult<PivotalRunnerLocal1<'_, PS::Native, SOP, N>>
    where
        N: Number,
        PS: ProbabilitySpec,
        SOP: PointSet<N>,
    {
        let controller = options.to_spreading_controller()?;
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<usize>::with_capacity(20);
        Ok(PivotalRunner {
            controller,
            strategy: Self {
                searcher,
                candidates,
            },
        })
    }
}
impl<ST, P, N> PivotalStrategy<SpreadingSampleController<'_, ST, N, P>> for LocalStrategy1<N>
where
    ST: ProbabilityStore,
    P: PointSet<N>,
    N: Number,
{
    fn select_pair<R>(
        &mut self,
        controller: &mut SpreadingSampleController<'_, ST, N, P>,
        rng: &mut R,
    ) -> Pair
    where
        R: RandomNumberGenerator,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        loop {
            let id1 = controller.indices().draw(rng).unwrap();
            self.searcher
                .reset_from_unit(controller.tree().data(), id1)
                .expect("id1 to exist")
                .search(controller.tree())
                .expect("nn to be found");
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend(self.searcher.neighbours().iter().map(|n| n.id()));

            {
                let mut i = 0usize;
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
                let id2 = *rng.relement(&self.candidates).unwrap();
                return Pair::More(id1, id2);
            }
        }
    }
}

pub type PivotalRunnerLocal1S<'b, PT, P, N> =
    PivotalRunner<SpreadingSampleController<'b, PT, N, P>, LocalStrategy1S<N>>;
pub struct LocalStrategy1S<N> {
    searcher: NearestNeighbourSearcher<N>,
    candidates: Vec<usize>,
    history: Vec<usize>,
}
impl<N> LocalStrategy1S<N> {
    pub fn new<PS, SOP, BOP>(
        options: &SamplingOptions<PS, SOP, BOP>,
    ) -> SamplingResult<PivotalRunnerLocal1S<'_, PS::Native, SOP, N>>
    where
        N: Number,
        PS: ProbabilitySpec,
        SOP: PointSet<N>,
    {
        let controller = options.to_spreading_controller()?;
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        let candidates = Vec::<usize>::with_capacity(20);
        let history = Vec::<usize>::with_capacity(controller.indices().len());
        Ok(PivotalRunner {
            controller,
            strategy: Self {
                searcher,
                candidates,
                history,
            },
        })
    }
}
impl<ST, P, N> PivotalStrategy<SpreadingSampleController<'_, ST, N, P>> for LocalStrategy1S<N>
where
    ST: ProbabilityStore,
    P: PointSet<N>,
    N: Number,
{
    fn select_pair<R>(
        &mut self,
        controller: &mut SpreadingSampleController<'_, ST, N, P>,
        rng: &mut R,
    ) -> Pair
    where
        R: RandomNumberGenerator,
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
            self.history.push(controller.indices().draw(rng).unwrap());
        }

        loop {
            let id1 = *self.history.last().unwrap();

            self.searcher
                .reset_from_unit(controller.tree().data(), id1)
                .expect("id1 to exist")
                .search(controller.tree())
                .expect("nn to be found");
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any of the potential
            // equals is a match
            self.candidates
                .extend(self.searcher.neighbours().iter().map(|n| n.id()));

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
                let id2 = *rng.relement(&self.candidates[0..left]).unwrap();
                return Pair::More(id1, id2);
            }
            // If no mutual nn has been found, we select one of the candidates by random to be the
            // next search unit ... but first we clear the history if it has become to long
            if self.history.len() == controller.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history.push(
                *rng.relement(&self.candidates)
                    .expect("candidates to have positive length"),
            );
        }
    }
}

pub type PivotalRunnerLocal2<'b, PT, P, N> =
    PivotalRunner<SpreadingSampleController<'b, PT, N, P>, LocalStrategy2<N>>;
pub struct LocalStrategy2<N> {
    pub(super) searcher: NearestNeighbourSearcher<N>,
}
impl<N> LocalStrategy2<N> {
    pub fn new<PS, SOP, BOP>(
        options: &SamplingOptions<PS, SOP, BOP>,
    ) -> SamplingResult<PivotalRunnerLocal2<'_, PS::Native, SOP, N>>
    where
        N: Number,
        PS: ProbabilitySpec,
        SOP: PointSet<N>,
    {
        let controller = options.to_spreading_controller()?;
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        Ok(PivotalRunner {
            controller,
            strategy: Self { searcher },
        })
    }
}
impl<ST, P, N> PivotalStrategy<SpreadingSampleController<'_, ST, N, P>> for LocalStrategy2<N>
where
    ST: ProbabilityStore,
    P: PointSet<N>,
    N: Number,
{
    fn select_pair<R>(
        &mut self,
        controller: &mut SpreadingSampleController<'_, ST, N, P>,
        rng: &mut R,
    ) -> Pair
    where
        R: RandomNumberGenerator,
    {
        let pair: Pair = controller.indices().into();
        if !pair.is_more() {
            return pair;
        }

        let id1 = controller.indices().draw(rng).unwrap();
        self.searcher
            .reset_from_unit(controller.tree().data(), id1)
            .expect("id1 to exist")
            .search(controller.tree())
            .expect("nn to be found");
        let id2 = rng
            .relement(self.searcher.neighbours())
            .expect("neighbours to have positive length")
            .id();

        Pair::More(id1, id2)
    }
}

pub trait LocalPivotalSampling<P, N>
where
    P: PointSet<N>,
    N: Number,
{
    fn lpm_1<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
    fn lpm_1s<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
    fn lpm_2<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
}
impl<PS, SOP, BOP, N> LocalPivotalSampling<SOP, N> for SamplingOptions<PS, SOP, BOP>
where
    PS: ProbabilitySpec,
    SOP: PointSet<N>,
    N: Number,
{
    /// Draw a sample using the local pivotal method 1.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_1(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
    /// Spatially balanced sampling through the pivotal method.
    /// Biometrics, 68(2), 514-520.
    /// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
    fn lpm_1<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(LocalStrategy1::new(self)?.sample(rng))
    }
    /// Draw a sample using the local pivotal method 1.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_1s(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
    /// Spatially balanced sampling through the pivotal method.
    /// Biometrics, 68(2), 514-520.
    /// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
    fn lpm_1s<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(LocalStrategy1S::new(self)?.sample(rng))
    }
    /// Draw a sample using the local pivotal method 2.
    /// The sample is spatially balanced on the provided auxilliary variables in `data`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_spreading(m)?
    ///     .lpm_2(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
    /// Spatially balanced sampling through the pivotal method.
    /// Biometrics, 68(2), 514-520.
    /// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
    fn lpm_2<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(LocalStrategy2::new(self)?.sample(rng))
    }
}
