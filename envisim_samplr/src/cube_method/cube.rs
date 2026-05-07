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

use std::num::NonZeroUsize;

use envisim_utils::kd_tree::searcher::{
    KNearestNeighbourSearcher,
    NeighbourSlice,
};
use envisim_utils::matrix::{
    Dimensions,
    Matrix,
    MatrixBase,
    MatrixDims,
    RawData,
};
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
    SpreadingOptions,
};
use envisim_utils::spatial::{
    Number,
    PointSet,
};

use super::utils::{
    find_vector_in_null_space,
    set_candidates_from_indices,
};
use crate::EqualProbabilitySampling;
use crate::error::SamplingResult;

pub trait CubeStrategy<C> {
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut C,
        rng: &mut R,
        n_units: usize,
    ) where
        R: RandomNumberGenerator;
    // Used for stratified
    fn reset_to_ids(&mut self, controller: &mut C, ids: &mut [usize], n_neighbours: usize);
}

pub struct CubeRunner<C, S> {
    pub(super) controller: C,
    pub(super) strategy: S,
    pub(super) candidates: Vec<usize>,
    pub(super) adjusted_data: Matrix<f64>,
    pub(super) candidate_data: Matrix<f64>,
}
impl<C, S> CubeRunner<C, S>
where
    C: SampleController<Store = FloatProbabilities>,
    S: CubeStrategy<C>,
{
    pub fn sample<R>(&mut self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        self.run(rng);
        self.controller.sample_mut().sort_to_vec()
    }
    pub fn run<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        self.run_flight(rng);
        self.run_landing(rng);
    }
    pub fn run_flight<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let b_cols = self.adjusted_data.ncol().get();
        assert_eq!(b_cols, self.candidate_data.nrow().get());

        while self.controller.indices().len() > b_cols {
            self.strategy
                .select_units(&mut self.candidates, &mut self.controller, rng, b_cols + 1);
            self.set_candidate_data();
            self.update_probabilities(rng);
        }
    }
    pub fn run_landing<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let b_cols = self.adjusted_data.ncol();
        let len = self.controller.indices().len();
        assert!(
            len <= b_cols.get(),
            "landing phase committed early: {len} units remaining, with {b_cols} cols",
        );

        while self.controller.indices().len() > 1 {
            set_candidates_from_indices(&mut self.candidates, &mut self.controller, 0);
            self.set_candidate_data();
            self.update_probabilities(rng)
        }

        self.controller
            .unit_decide_last(rng)
            .expect("last unit to be decided");
    }
    pub fn new<PS, SOP, T>(
        options: &SamplingOptions<PS, SOP, MatrixBase<T>>,
        controller: C,
        strategy: S,
    ) -> SamplingResult<Self>
    where
        PS: ProbabilitySpec,
        T: RawData<Elem = f64>,
    {
        let balancing_data = options.balancing()?.data();
        let b_dims = balancing_data.dims();
        let mut adjusted_data = balancing_data.to_matrix();

        for i in 0..b_dims.rows.get() {
            let p = controller.probabilities().get(i);
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

        Ok(Self {
            controller,
            strategy,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data,
            candidate_data,
        })
    }
    fn set_candidate_data(&mut self) {
        let n_candidates = self.candidates.len();
        assert!(n_candidates <= self.adjusted_data.ncol().get() + 1);
        let dims = MatrixDims::try_new(n_candidates - 1, n_candidates).expect("n_candidates > 1");
        self.candidate_data.resize(dims);

        for (i, &id) in self.candidates.iter().enumerate() {
            for j in 0..dims.rows.get() {
                self.candidate_data[(j, i)] = self.adjusted_data[(id, j)];
            }
        }
    }
    fn update_probabilities<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let uvec = find_vector_in_null_space(&mut self.candidate_data);
        let mut lambdas = (f64::MAX, f64::MAX);

        for (prob, &uval) in self
            .candidates
            .iter()
            .map(|&id| self.controller.probabilities().get(id))
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

        let lambda = if rng.one_of_f64(lambdas.0, lambdas.1).unwrap() {
            lambdas.0
        } else {
            -lambdas.1
        };

        for (i, &id) in self.candidates.iter().enumerate() {
            self.controller
                .unit_add_and_decide(id, lambda * uvec[i])
                .expect("id to update");
        }
    }
}

pub struct BasicCubeStrategy();
impl BasicCubeStrategy {
    pub fn new<PS, SOP, T>(
        options: &SamplingOptions<PS, SOP, MatrixBase<T>>,
    ) -> SamplingResult<CubeRunner<BasicSampleController<FloatProbabilities>, Self>>
    where
        PS: ProbabilitySpec,
        T: RawData<Elem = f64>,
    {
        let controller = options.to_controller_float();
        CubeRunner::new(options, controller, BasicCubeStrategy())
    }
}
impl<C> CubeStrategy<C> for BasicCubeStrategy
where
    C: SampleController<Store = FloatProbabilities>,
{
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut C,
        _rng: &mut R,
        n_units: usize,
    ) where
        R: RandomNumberGenerator,
    {
        set_candidates_from_indices(candidates, controller, n_units)
    }
    fn reset_to_ids(&mut self, controller: &mut C, ids: &mut [usize], _n_neighbours: usize) {
        controller.indices_mut().clear();
        for &id in ids.iter() {
            controller.indices_mut().insert(id).unwrap();
        }
    }
}

pub struct LocalCubeStrategy<'a, P, N> {
    pub(super) spreading_options: &'a SpreadingOptions<P>,
    pub(super) searcher: KNearestNeighbourSearcher<N>,
}
impl<'a, SOP, N> LocalCubeStrategy<'a, SOP, N> {
    pub fn new<PS, T>(
        options: &'a SamplingOptions<PS, SOP, MatrixBase<T>>,
    ) -> SamplingResult<CubeRunner<SpreadingSampleController<'a, FloatProbabilities, N, SOP>, Self>>
    where
        SOP: PointSet<N>,
        N: Number,
        PS: ProbabilitySpec,
        T: RawData<Elem = f64>,
    {
        let controller = options.to_spreading_controller_float()?;
        let searcher = KNearestNeighbourSearcher::new(
            options.balancing()?.data().ncol(),
            controller.tree().data(),
        );
        CubeRunner::new(
            options,
            controller,
            Self {
                spreading_options: options.spreading()?,
                searcher,
            },
        )
    }
}
impl<'a, P, N> CubeStrategy<SpreadingSampleController<'a, FloatProbabilities, N, P>>
    for LocalCubeStrategy<'a, P, N>
where
    P: PointSet<N>,
    N: Number,
{
    fn select_units<R>(
        &mut self,
        candidates: &mut Vec<usize>,
        controller: &mut SpreadingSampleController<'a, FloatProbabilities, N, P>,
        rng: &mut R,
        n_units: usize,
    ) where
        R: RandomNumberGenerator,
    {
        assert!(n_units > 1);
        let len = controller.indices().len();
        assert!(len >= n_units);

        if len == n_units {
            return set_candidates_from_indices(candidates, controller, n_units);
        }

        candidates.clear();

        // Draw the first unit at random
        let id1 = controller.indices().draw(rng).unwrap();
        candidates.push(id1);

        // Find the neighbours of this first unit
        self.searcher
            .reset_from_unit(controller.tree().data(), id1)
            .expect("id1 to exist")
            .search(controller.tree())
            .expect("nn to be found");

        // Add all neighbours, if no equals
        if self.searcher.neighbours().len() == n_units - 1 {
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
        let n_open_spots = n_units - candidates.len();
        let opts = SamplingOptions::new_equal(n_remaining_units, n_open_spots).unwrap();

        let s = opts.srs(rng);
        for k in s {
            candidates.push(self.searcher.neighbours()[guaranteed_units + k].id());
        }
    }
    fn reset_to_ids(
        &mut self,
        controller: &mut SpreadingSampleController<'a, FloatProbabilities, N, P>,
        ids: &mut [usize],
        n_neighbours: usize,
    ) {
        self.searcher.set_nominal_size(
            NonZeroUsize::new(n_neighbours).expect("n_neighbours to be positive"),
        );

        controller.indices_mut().clear();
        controller
            .reset_tree(self.spreading_options, ids)
            .expect("tree should be resettable");

        for id in ids.iter() {
            controller.indices_mut().insert(*id).unwrap();
        }
    }
}

pub trait CubeSampling {
    fn cube<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
}
pub trait LocalCubeSampling<P, N>
where
    P: PointSet<N>,
    N: Number,
{
    fn local_cube<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator;
}
impl<PS, SOP, T> CubeSampling for SamplingOptions<PS, SOP, MatrixBase<T>>
where
    PS: ProbabilitySpec,
    T: RawData<Elem = f64>,
{
    /// Draw a sample using the cube method.
    /// The sample is balanced on the provided auxilliary variables in `balancing`.
    /// For fixed sized samples, the first auxilliary variable should be the probability vector.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_balancing(m)?
    ///     .cube(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tillé, Y. (2004).
    /// Efficient balanced sampling: the cube method.
    /// Biometrika, 91(4), 893-912.
    /// <https://doi.org/10.1093/biomet/91.4.893>
    fn cube<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(BasicCubeStrategy::new(self)?.sample(rng))
    }
}
impl<PS, SOP, N, T> LocalCubeSampling<SOP, N> for SamplingOptions<PS, SOP, MatrixBase<T>>
where
    PS: ProbabilitySpec,
    SOP: PointSet<N>,
    N: Number,
    T: RawData<Elem = f64>,
{
    /// Draw a sample using the local cube method.
    /// The sample is balanced on the provided auxilliary variables in `balancing`.
    /// the sample is spatially balanced on the provided auxilliary variables in `auxiliaries`.
    /// For fixed sized samples, the first auxilliary variable should be the probability vector.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = SmallRng::from_os_rng();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let bal = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let spr = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p.into())?
    ///     .set_balancing(bal)?
    ///     .set_spreading(spr)?
    ///     .local_cube(&mut rng)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Deville, J. C., & Tillé, Y. (2004).
    /// Efficient balanced sampling: the cube method.
    /// Biometrika, 91(4), 893-912.
    /// <https://doi.org/10.1093/biomet/91.4.893>
    ///
    /// Grafström, A., & Tillé, Y. (2013).
    /// Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
    /// Environmetrics, 24(2), 120-131.
    /// <https://doi.org/10.1002/env.2194>
    fn local_cube<R>(&self, rng: &mut R) -> SamplingResult<Vec<usize>>
    where
        R: RandomNumberGenerator,
    {
        Ok(LocalCubeStrategy::new(self)?.sample(rng))
    }
}
