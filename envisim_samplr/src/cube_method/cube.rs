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

//! Standard cube methods

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::searcher::{
    KNearestNeighbourSearcher,
    NeighbourSlice,
};
use envisim_utils::kd_tree::{
    PointSet,
    Tree,
};
use envisim_utils::matrix::{
    Dimensions,
    Matrix,
    MatrixBase,
    MatrixDims,
    RawData,
};
use envisim_utils::random::{
    FloatRng,
    Rand,
    Rng,
    random_weighted,
};
use envisim_utils::sample_controller::{
    SampleController,
    UnitRemoving,
};
use envisim_utils::sampling_options::{
    BalancingOptions,
    ProbabilityOptions,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};

use super::utils::{
    find_vector_in_null_space,
    set_candidates_from_indices_randomly,
    set_candidates_from_indices_sequentially,
};
use crate::EqualProbabilitySampling;

pub trait CubeStrategy<TREE> {
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut SampleController<f64, TREE>,
        rng: &mut R,
        n_units: NonZeroUsize,
    ) where
        R: Rand<usize>;
    // Used for stratified
    fn reset_to_ids(
        &mut self,
        controller: &mut SampleController<f64, TREE>,
        ids: &mut [usize],
        n_neighbours: usize,
    );
}

#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "super is ok, needed for stratified cube"
)]
#[must_use]
pub struct CubeRunner<S, TREE> {
    /// Sample controller
    pub(super) controller: SampleController<f64, TREE>,
    /// Sample strategy
    pub(super) strategy: S,
    /// Candidates for balancing
    pub(super) candidates: Vec<usize>,
    /// Probability-adjusted balancing data
    pub(super) adjusted_data: Matrix<f64>,
    /// Balancing data for candidates
    pub(super) candidate_data: Matrix<f64>,
}
impl<S, TREE> CubeRunner<S, TREE>
where
    S: CubeStrategy<TREE>,
    SampleController<f64, TREE>: UnitRemoving,
{
    /// Runs the simulation and returns a sorted sample
    #[must_use]
    #[inline]
    pub fn sample<R>(mut self, rng: &mut R) -> Vec<usize>
    where
        R: FloatRng,
    {
        self.run(rng);
        self.controller.to_sorted_sample_vec()
    }
    /// Runs the sampling algorithm
    #[inline]
    pub fn run<R>(&mut self, rng: &mut R)
    where
        R: FloatRng,
    {
        self.run_flight(rng);
        self.run_landing(rng);
    }
    /// Runs the flight phase of the cube method
    ///
    /// # Panics
    /// Panics if the width of `adjusted_data` does not match the `candidate_data` size (nrows), as
    /// this is a sign that all flight preparations were not done correctly.
    #[inline]
    pub fn run_flight<R>(&mut self, rng: &mut R)
    where
        R: FloatRng,
    {
        let b_cols = self.adjusted_data.ncol();
        assert_eq!(
            b_cols,
            self.candidate_data.nrow(),
            "flight phase not setup properly"
        );

        while self.controller.indices().len() > b_cols.get() {
            self.strategy.select_units(
                &mut self.candidates,
                &mut self.controller,
                rng,
                b_cols.saturating_add(1),
            );
            self.set_candidate_data();
            self.update_probabilities(rng);
        }
    }
    /// Runs the landing phase of the cube method
    ///
    /// # Panics
    /// Crashes if the landing phase is started too early, i.e. if the width of the `adjusted_data`
    /// is lower than the number of remaining units.
    #[inline]
    pub fn run_landing<R>(&mut self, rng: &mut R)
    where
        R: FloatRng,
    {
        let b_cols = self.adjusted_data.ncol();
        let len = self.controller.indices().len();
        assert!(
            len <= b_cols.get(),
            "landing phase committed early: {len} units remaining, with {b_cols} cols",
        );

        while self.controller.indices().len() > 1 {
            set_candidates_from_indices_sequentially(
                &mut self.candidates,
                self.controller.indices(),
                NonZeroUsize::MAX,
            );
            self.set_candidate_data();
            self.update_probabilities(rng);
        }

        self.controller.unit_decide_last(rng);
    }
    /// Constructs a new cube runner
    #[expect(
        clippy::missing_panics_doc,
        reason = "safe to assume balancing dims is not pushing the usize limit"
    )]
    #[inline]
    pub fn new<PO, AUX, T>(
        options: &SamplingOptions<PO, AUX, BalancingOptions<MatrixBase<T>>>,
        controller: SampleController<f64, TREE>,
        strategy: S,
    ) -> Self
    where
        PO: ProbabilityOptions<Real = f64>,
        T: RawData<Elem = f64>,
    {
        let balancing_data = options.balancing().data();
        let b_dims = balancing_data.dims();
        let mut adjusted_data = balancing_data.to_matrix();

        for i in 0..b_dims.rows.get() {
            let p = controller.probabilities()[i].get();
            for j in 0..b_dims.cols.get() {
                adjusted_data[(i, j)] /= p;
            }
        }

        let c_dims = MatrixDims::new(
            b_dims.cols,
            b_dims
                .cols
                .checked_add(1)
                .expect("b_dims to not overflow by adding 1"),
        );
        let candidate_data = Matrix::from_value(0.0, c_dims);

        Self {
            controller,
            strategy,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data,
            candidate_data,
        }
    }
    /// Sets the candidate data according to the candidates
    ///
    /// # Panics
    /// Panics if the number of candidates are too few, many or too few.
    #[inline]
    fn set_candidate_data(&mut self) {
        let n_candidates = self.candidates.len();
        assert!(
            n_candidates <= self.adjusted_data.ncol().get() + 1,
            "number of candidates needs to be at most one more than the width of adj data"
        );
        let dims = MatrixDims::try_new(n_candidates - 1, n_candidates).expect("n_candidates > 1");
        self.candidate_data.resize(dims);

        for (i, &id) in self.candidates.iter().enumerate() {
            for j in 0..dims.rows.get() {
                self.candidate_data[(j, i)] = self.adjusted_data[(id, j)];
            }
        }
    }
    /// Updates the probabilities of the candidates
    #[inline]
    fn update_probabilities<R>(&mut self, rng: &mut R)
    where
        R: Rand<f64>,
    {
        let uvec = find_vector_in_null_space(&mut self.candidate_data);
        let mut lambdas = (f64::MAX, f64::MAX);

        for (prob, &uval) in self
            .candidates
            .iter()
            .map(|&id| self.controller.probabilities()[id].get())
            .zip(uvec.iter())
        {
            let lvals = ((prob / uval).abs(), ((1.0 - prob) / uval).abs());

            if uval >= 0.0 {
                lambdas.0 = lambdas.0.min(lvals.1);
                lambdas.1 = lambdas.1.min(lvals.0);
            } else {
                lambdas.0 = lambdas.0.min(lvals.0);
                lambdas.1 = lambdas.1.min(lvals.1);
            }
        }

        let lambda = if random_weighted(rng, lambdas.0, lambdas.1)
            .expect("both lambdas to be non-negative, with at least one positive")
        {
            lambdas.0
        } else {
            -lambdas.1
        };

        for (i, &id) in self.candidates.iter().enumerate() {
            let delta = lambda * uvec[i];
            let _prest = self.controller.unit_add_delta_and_decide(id, delta);
        }
    }
}

/// Sequential cube strategy
#[must_use]
pub struct SequentialCubeStrategy();
impl SequentialCubeStrategy {
    /// Constructs a new cube runner using the sequential cube strategy
    #[inline]
    pub fn new<PO, AUX, T>(
        options: &SamplingOptions<PO, AUX, BalancingOptions<MatrixBase<T>>>,
    ) -> CubeRunner<Self, ()>
    where
        PO: ProbabilityOptions<Real = f64>,
        T: RawData<Elem = f64>,
    {
        let controller = options.to_controller_real();
        CubeRunner::new(options, controller, Self())
    }
}
impl CubeStrategy<()> for SequentialCubeStrategy {
    #[inline]
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut SampleController<f64, ()>,
        _rng: &mut R,
        n_units: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        set_candidates_from_indices_sequentially(candidates, controller.indices(), n_units);
    }
    /// # Panics
    /// If the id alread exists in the collection. This implies that the provided `ids` contains
    /// duplicates.
    #[inline]
    fn reset_to_ids(
        &mut self,
        controller: &mut SampleController<f64, ()>,
        ids: &mut [usize],
        _n_neighbours: usize,
    ) {
        controller.indices_mut().clear();
        for &id in ids.iter() {
            controller.indices_mut().insert(id);
        }
    }
}

/// Random cube strategy
#[must_use]
pub struct RandomCubeStrategy();
impl RandomCubeStrategy {
    /// Constructs a new cube runner using the random cube strategy
    #[inline]
    pub fn new<PO, AUX, T>(
        options: &SamplingOptions<PO, AUX, BalancingOptions<MatrixBase<T>>>,
    ) -> CubeRunner<Self, ()>
    where
        PO: ProbabilityOptions<Real = f64>,
        T: RawData<Elem = f64>,
    {
        let controller = options.to_controller_real();
        CubeRunner::new(options, controller, Self())
    }
}
impl CubeStrategy<()> for RandomCubeStrategy {
    #[inline]
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut SampleController<f64, ()>,
        rng: &mut R,
        n_units: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        set_candidates_from_indices_randomly(rng, candidates, controller.indices(), n_units);
    }
    /// # Panics
    /// If the id alread exists in the collection. This implies that the provided `ids` contains
    /// duplicates.
    #[inline]
    fn reset_to_ids(
        &mut self,
        controller: &mut SampleController<f64, ()>,
        ids: &mut [usize],
        _n_neighbours: usize,
    ) {
        controller.indices_mut().clear();
        for &id in ids.iter() {
            controller.indices_mut().insert(id);
        }
    }
}

#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "super is ok, needed for stratified cube"
)]
#[must_use]
pub struct LocalCubeStrategy<'btree, P>
where
    P: PointSet,
{
    /// The spreading options, needed in order to reset the tree (used in stratified cube)
    pub(super) spreading_options: &'btree SpreadingOptions<P>,
    /// Searcher, used in tree
    pub(super) searcher: KNearestNeighbourSearcher<P>,
}
impl<'btree, P> LocalCubeStrategy<'btree, P>
where
    P: PointSet<Id = usize>,
{
    /// Constructs a new cube runner using the local cube strategy
    #[inline]
    pub fn new<PO, T>(
        options: &'btree SamplingOptions<PO, SpreadingOptions<P>, BalancingOptions<MatrixBase<T>>>,
    ) -> CubeRunner<Self, Tree<'btree, P>>
    where
        PO: ProbabilityOptions<Real = f64>,
        T: RawData<Elem = f64>,
    {
        let controller = options.to_spreading_controller_real();
        let searcher = KNearestNeighbourSearcher::new(
            options.balancing().data().ncol(),
            controller.tree().data(),
        );
        CubeRunner::new(
            options,
            controller,
            Self {
                spreading_options: options.spreading(),
                searcher,
            },
        )
    }
}
impl<'btree, P> CubeStrategy<Tree<'btree, P>> for LocalCubeStrategy<'btree, P>
where
    P: PointSet<Id = usize>,
{
    /// # Panics
    /// Panics if only one unit is wanted, or if more units is wanted than remains.
    #[inline]
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut SampleController<f64, Tree<'btree, P>>,
        rng: &mut R,
        n_units: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        assert!(
            n_units.get() > 1,
            "only one unit wanted, should have entered landing phase"
        );
        let len = controller.indices().len();

        if len <= n_units.get() {
            return set_candidates_from_indices_sequentially(
                candidates,
                controller.indices(),
                n_units,
            );
        }

        candidates.clear();

        // Draw the first unit at random
        let id1 = controller
            .indices()
            .draw(rng)
            .expect("we have already concluded that more that one unit remains");
        candidates.push(id1);

        // Find the neighbours of this first unit
        self.searcher
            .reset_from_unit(controller.tree().data(), id1)
            .expect("id1 to exist")
            .search(controller.tree())
            .expect("nn to be found");

        // Add all neighbours, if no equals
        if self.searcher.neighbours().len() == n_units.get() - 1 {
            candidates.extend(self.searcher.neighbours().to_neighbour_id_iter());
            return;
        }

        // There exists multiple max_distance neighbours, we need to add the non max, and then
        // sample amongst the max'es
        let max_distance = self
            .searcher
            .max_distance()
            .expect("searcher to have found a neighbour");
        // Units with distance below max_distance are guaranteed their weight
        let guaranteed_units = self
            .searcher
            .neighbours()
            .partition_point(|n| n.distance() < max_distance);
        candidates.extend(self.searcher.neighbours()[0..guaranteed_units].to_neighbour_id_iter());

        // Randomly add neighbours on the maximum distance
        // We need to draw from the
        let n_remaining_units =
            NonZeroUsize::new(self.searcher.neighbours().len() - guaranteed_units)
                .expect("more than one unit to remain on the border");
        // the number left to fill amongst the candidates
        let n_open_spots = n_units.get() - candidates.len();
        let opts = SamplingOptions::new_equal(n_remaining_units, n_open_spots)
            .expect("n_remaining_units to be larger than n_open_spots");

        let s = opts.srs(rng);
        for k in s {
            candidates.push(self.searcher.neighbours()[guaranteed_units + k].id());
        }
    }
    /// # Panics
    /// If the id alread exists in the collection. This implies that the provided `ids` contains
    /// duplicates.
    #[inline]
    fn reset_to_ids(
        &mut self,
        controller: &mut SampleController<f64, Tree<'btree, P>>,
        ids: &mut [usize],
        n_neighbours: usize,
    ) {
        self.searcher.set_nominal_size(
            NonZeroUsize::new(n_neighbours).expect("n_neighbours to be positive"),
        );

        controller.indices_mut().clear();
        controller.reset_tree(self.spreading_options, ids);

        for id in ids.iter() {
            controller.indices_mut().insert(*id);
        }
    }
}

pub trait CubeSampling<R>
where
    R: Rng,
{
    /// Draw a sample using the cube method.
    /// The sample is balanced on the provided auxilliary variables in `balancing`.
    /// For fixed sized samples, the first auxilliary variable should be the probability vector.
    ///
    /// Units are selected randomly to the flight phase.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_balancing(m)?
    ///     .cube(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn cube(&self, rng: &mut R) -> Vec<usize>;
    /// Draw a sample using the cube method.
    /// The sample is balanced on the provided auxilliary variables in `balancing`.
    /// For fixed sized samples, the first auxilliary variable should be the probability vector.
    ///
    /// Units are selected in sequence to the flight phase.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_balancing(m)?
    ///     .cube(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    #[must_use]
    fn sequential_cube(&self, rng: &mut R) -> Vec<usize>;
}
pub trait LocalCubeSampling<R>
where
    R: Rng,
{
    #[must_use]
    fn local_cube(&self, rng: &mut R) -> Vec<usize>;
}
impl<R, PO, AUX, T> CubeSampling<R> for SamplingOptions<PO, AUX, BalancingOptions<MatrixBase<T>>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions<Real = f64>,
    T: RawData<Elem = f64>,
{
    #[inline]
    fn cube(&self, rng: &mut R) -> Vec<usize> { RandomCubeStrategy::new(self).sample(rng) }
    #[inline]
    fn sequential_cube(&self, rng: &mut R) -> Vec<usize> {
        SequentialCubeStrategy::new(self).sample(rng)
    }
}
impl<R, PO, P, T> LocalCubeSampling<R>
    for SamplingOptions<PO, SpreadingOptions<P>, BalancingOptions<MatrixBase<T>>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize>,
    T: RawData<Elem = f64>,
{
    #[inline]
    fn local_cube(&self, rng: &mut R) -> Vec<usize> { LocalCubeStrategy::new(self).sample(rng) }
}
