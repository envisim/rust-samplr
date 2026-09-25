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

use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::{
    KNearestNeighbourSearcher,
    Neighbour,
    NeighbourView,
};
use envisim_utils::matrix::{
    Dimensions,
    Matrix,
    MatrixDims,
};
use envisim_utils::probabilities::{
    Probability,
    ProbabilitySet,
    ProbabilityStore,
};
use envisim_utils::random::{
    FloatRng,
    Rand,
    Rng,
    random_weighted,
};
use envisim_utils::sample_controller::{
    SampleController,
    TreeStorage,
};
use envisim_utils::sampling_options::{
    BaseProbabilitiesSpec,
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::{
    DataViewMut,
    PointSet,
};

use super::utils::{
    CubeError,
    find_vector_in_null_space,
};
use crate::equal::EqualProbabilitySampling;

/// Cube method runner.
pub struct CubeMethod<'bopts, PS, AUX, BL, PR, TR, SE>
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
{
    /// Original options object.
    pub options: &'bopts SamplingOptions<PS, AUX>,
    /// Balancing data.
    pub balancing: BL,
    /// Controller.
    pub controller: SampleController<PR, TR>,
    /// Possible searcher (for spatial cube).
    pub searcher: SE,
    /// List of candidates.
    pub candidates: Vec<PS::Id>,
    /// Probability-adjusted balancing data, transposed. Should have size list.len-1 x list.len,
    /// or rather, the number of candidates should not be larger than one less than the number of
    /// balancing dims.
    pub cand_data: Matrix<f64>,
}
impl<'bopts, PS, AUX, BL, PR, TR, SE> CubeMethod<'bopts, PS, AUX, BL, PR, TR, SE>
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
    TR: TreeStorage<PS::Id>,
{
    /// Set candidate data from candidates, standard method. First column need to be probabiliites
    /// to ensure fixed sized sample.
    /// # Panics
    /// Panics if candidate list is not long enough
    #[inline]
    fn set_data(&mut self) {
        assert!(
            !self.candidates.len() >= 2,
            "candidates list must have len >= 2"
        );
        assert!(
            self.candidates.len() - 1 <= self.cand_data.ncol().get(),
            "canidadets list-1 must be smaller than width of balancing data"
        );
        let dims = MatrixDims::try_new(self.candidates.len() - 1, self.candidates.len())
            .expect("len >= 2");
        self.cand_data.resize(dims);

        for (i, &id) in self.candidates.iter().enumerate() {
            let p = self
                .options
                .probabilities()
                .get_real(id)
                .expect("id to exist in probabilities");
            for j in 0..dims.rows.get() {
                self.cand_data[(j, i)] = self
                    .balancing
                    .coord(id, j)
                    .expect("id, j to exist in balancing")
                    / p;
            }
        }
    }
    /// Set candidates sequentially from the list
    #[inline]
    pub fn set_candidates_sequentially(&mut self, len: NonZeroUsize) {
        self.candidates.clear();
        if self.controller.indices().len() <= len.get() {
            // Few units remaining, select everything
            self.candidates
                .extend_from_slice(self.controller.indices().list());
        } else {
            self.candidates.extend(
                self.controller
                    .probabilities()
                    .ids()
                    .filter(|id| self.controller.indices().contains(*id))
                    .take(len.get()),
            );
        }
    }
    /// Update probabilities. Requires candidates to have been selected, and data to have been set.
    #[inline]
    pub fn update_probabilities<R>(&mut self, rng: &mut R)
    where
        R: Rand<f64>,
    {
        let uvec = self.find_vector_in_null_space();
        let mut lambdas = (f64::MAX, f64::MAX);

        for (prob, &uval) in self
            .candidates
            .iter()
            .map(|&id| {
                self.controller
                    .probabilities()
                    .get(id)
                    .expect("id to exist")
                    .get()
            })
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
    /// Find vector in null space of candidate data
    /// # Panics
    /// Panics if the candidate data matrix is not n-1 x n
    #[must_use]
    #[inline]
    fn find_vector_in_null_space(&mut self) -> Vec<f64> {
        find_vector_in_null_space(&mut self.cand_data)
    }
    /// Run the flight phase
    /// # Panics
    /// Panics if flight phase is not set up properly, i.e. candidate data shape is incorrect
    #[inline]
    fn run_flight<R, S>(&mut self, rng: &mut R, strategy: S)
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        let b_cols = self.balancing.dimensions();
        let c_cols = b_cols; // Assume no prob column
        let n_candidates = c_cols.checked_add(1).expect("no overflow"); // Select one more than cols

        assert_eq!(
            self.cand_data.dims(),
            (c_cols, n_candidates).into(),
            "flight phase not setup properly"
        );

        while self.controller.indices().len() >= n_candidates.get() {
            strategy.select_units(rng, self, c_cols);
            self.set_data();
            self.update_probabilities(rng);
        }
    }
    /// Run the landing phase
    /// # Panics
    /// Panics if landing is committed too early
    #[inline]
    fn run_landing<R>(&mut self, rng: &mut R)
    where
        R: FloatRng,
    {
        let b_cols = self.balancing.dimensions();
        let len = self.controller.indices().len();
        assert!(
            len <= b_cols.get(),
            "landing phase committed early: {len} units remaining, with {b_cols} cols",
        );

        while self.controller.indices().len() > 1 {
            self.set_candidates_sequentially(NonZeroUsize::MAX);
            self.set_data();
            self.update_probabilities(rng);
        }

        self.controller.unit_decide_last(rng);
    }
    /// Run the cube algortihm and return a sample
    #[inline]
    fn sample<R, S>(mut self, rng: &mut R, strategy: S) -> Self
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        self.run_flight(rng, strategy);
        self.run_landing(rng);
        self
    }
}
impl<'bopts, PS, AUX, BL, T> CubeMethod<'bopts, PS, AUX, BL, ProbabilitySet<T, f64>, (), ()>
where
    PS: ProbabilitiesSpec<Real = f64, ConstructableContainer<Probability<f64>> = T>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    T: DataViewMut<Id = PS::Id, Value = Probability<f64>>,
{
    /// Constructs a new, non-spatial, runner
    /// # Panics
    /// Panics if balancing dims tend to overflow in small additions
    #[inline]
    pub fn new(
        options: &'bopts SamplingOptions<PS, AUX>,
        balancing: BL,
    ) -> Result<Self, CubeError> {
        CubeError::check_balancing(options.probabilities(), &balancing)?;
        let b_dims = balancing.dimensions();
        let c_dims = MatrixDims::new(
            b_dims,
            b_dims
                .checked_add(1)
                .expect("b_dims to not overflow by adding 1"),
        );

        let cand_data = Matrix::from_value(0.0, c_dims);
        Ok(CubeMethod {
            controller: SampleController::new_real(options),
            options,
            balancing,
            searcher: (),
            candidates: Vec::<PS::Id>::with_capacity(b_dims.get()),
            cand_data,
        })
    }
}
impl<'bopts, PS, BL, T, P>
    CubeMethod<
        'bopts,
        PS,
        SpreadingOptions<P>,
        BL,
        ProbabilitySet<T, f64>,
        Tree<'bopts, P>,
        KNearestNeighbourSearcher<P>,
    >
where
    PS: ProbabilitiesSpec<Real = f64, ConstructableContainer<Probability<f64>> = T>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    T: DataViewMut<Id = PS::Id, Value = Probability<f64>>,
    P: PointSet<Id = PS::Id>,
{
    /// Constructs a new spatial runner
    #[inline]
    pub fn new_spreading(
        options: &'bopts SamplingOptions<PS, SpreadingOptions<P>>,
        balancing: BL,
    ) -> Result<Self, CubeError> {
        let c = CubeMethod::new(options, balancing)?;
        let searcher =
            KNearestNeighbourSearcher::new(c.balancing.dimensions(), options.spreading().data());
        Ok(CubeMethod {
            controller: SampleController::new_real_spreading(options),
            options,
            balancing: c.balancing,
            searcher,
            candidates: c.candidates,
            cand_data: c.cand_data,
        })
    }
}

/// A strategy for CUBE controls the selection of the units.
pub trait CubeStrategy<'bopts, PS, AUX, BL, PR>: Copy
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
{
    /// Possible tree type for spatial cube (or void)
    type Tree;
    /// Possible searcher type for spatial cube (or void)
    type Searcher;

    /// Selects a subset of units to be used for a step of the algorithm
    fn select_units<R>(
        self,
        rng: &mut R,
        cube: &mut CubeMethod<'bopts, PS, AUX, BL, PR, Self::Tree, Self::Searcher>,
        len: NonZeroUsize,
    ) where
        R: Rand<usize>;
    /// Resets the ids of the index controller. Used for stratified CUBE.
    #[inline]
    fn reset_to_ids(
        self,
        cube: &mut CubeMethod<'bopts, PS, AUX, BL, PR, Self::Tree, Self::Searcher>,
        ids: &mut [PS::Id],
    ) {
        cube.controller.indices_mut().clear();
        for id in ids {
            cube.controller.indices_mut().insert(*id);
        }
    }
}

/// Sequential cube strategy
#[derive(Clone, Copy)]
pub struct SequentialStrategy;
impl<PS, AUX, BL, PR> CubeStrategy<'_, PS, AUX, BL, PR> for SequentialStrategy
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
{
    type Tree = ();
    type Searcher = ();
    #[inline]
    fn select_units<R>(
        self,
        _rng: &mut R,
        cube: &mut CubeMethod<'_, PS, AUX, BL, PR, Self::Tree, Self::Searcher>,
        len: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        cube.set_candidates_sequentially(len);
    }
}

/// Random cube strategy
#[derive(Clone, Copy)]
pub struct RandomStrategy;
impl<PS, AUX, BL, PR> CubeStrategy<'_, PS, AUX, BL, PR> for RandomStrategy
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
{
    type Tree = ();
    type Searcher = ();
    #[inline]
    fn select_units<R>(
        self,
        rng: &mut R,
        cube: &mut CubeMethod<'_, PS, AUX, BL, PR, Self::Tree, Self::Searcher>,
        len: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        cube.candidates.clear();
        if cube.controller.indices().len() < len.get() {
            // Few units remaining, select everything
            cube.candidates
                .extend_from_slice(cube.controller.indices().list());
        } else {
            // Draw an srs
            cube.candidates.extend(
                SamplingOptions::new_equal(cube.controller.indices().len(), len.get())
                    .expect("indices.len > 0")
                    .srs(rng)
                    .iter()
                    .map(|&k| cube.controller.indices()[k]),
            );
        }
    }
}

/// Local cube strategy
#[derive(Clone, Copy)]
pub struct SpatialStrategy;
impl<'bopts, PS, P, BL, PR> CubeStrategy<'bopts, PS, SpreadingOptions<P>, BL, PR>
    for SpatialStrategy
where
    PS: BaseProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PS::Id> + 'bopts,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
{
    type Tree = Tree<'bopts, P>;
    type Searcher = KNearestNeighbourSearcher<P>;
    #[inline]
    fn select_units<R>(
        self,
        rng: &mut R,
        cube: &mut CubeMethod<'bopts, PS, SpreadingOptions<P>, BL, PR, Self::Tree, Self::Searcher>,
        len: NonZeroUsize,
    ) where
        R: Rand<usize>,
    {
        cube.candidates.clear();
        if cube.controller.indices().len() < len.get() {
            // Few units remaining, select everything
            cube.candidates
                .extend_from_slice(cube.controller.indices().list());
            return;
        }

        // Draw the first unit at random
        let id1 = cube
            .controller
            .indices()
            .draw(rng)
            .expect("indices contains units");
        cube.candidates.push(id1);

        // Find the neighbours of this first unit
        cube.searcher
            .set_nominal_size(len)
            .reset_from_unit(cube.controller.tree().data(), id1)
            .expect("id1 to exist")
            .search(cube.controller.tree())
            .expect("nn to be found");

        // Add all neighbours, if no equals
        if cube.searcher.neighbours().len() == len.get() - 1 {
            cube.candidates
                .extend(cube.searcher.neighbours().iter().map(Neighbour::id));
            return;
        }

        // There exists multiple max_distance neighbours, we need to add the non max, and then
        // sample amongst the max'es
        let max_distance = cube
            .searcher
            .max_distance()
            .expect("searcher to have found a neighbour");
        // Units with distance below max_distance are guaranteed their weight
        let guaranteed_units = cube
            .searcher
            .neighbours()
            .partition_point(|n| n.distance() < max_distance);
        cube.candidates.extend(
            cube.searcher.neighbours()[0..guaranteed_units]
                .iter()
                .map(Neighbour::id),
        );

        // Randomly add neighbours on the maximum distance
        // We need to draw from the
        let n_remaining_units =
            NonZeroUsize::new(cube.searcher.neighbours().len() - guaranteed_units)
                .expect("more than one unit to remain on the border");
        // the number left to fill amongst the candidates
        let n_open_spots = len.get() - cube.candidates.len();
        let opts = SamplingOptions::new_equal(n_remaining_units, n_open_spots)
            .expect("n_remaining_units to be larger than n_open_spots");

        let s = opts.srs(rng);
        for k in s {
            cube.candidates
                .push(*cube.searcher.neighbours()[guaranteed_units + k].id());
        }
    }
    #[inline]
    fn reset_to_ids(
        self,
        cube: &mut CubeMethod<'bopts, PS, SpreadingOptions<P>, BL, PR, Self::Tree, Self::Searcher>,
        ids: &mut [PS::Id],
    ) {
        cube.controller.indices_mut().clear();
        for id in ids.iter() {
            cube.controller.indices_mut().insert(*id);
        }

        *cube.controller.tree_mut() =
            Tree::new(cube.options.spreading(), ids).expect("ids to exist in data");
    }
}

/// Provides CUBE sampling methods
pub trait CubeSampling<ID, R>
where
    R: Rng,
{
    /// Draw a sample using the cube method.
    /// The sample is balanced on the provided auxiliary variables in `balancing`.
    /// For fixed sized samples, the first auxiliary variable should be the probability vector.
    ///
    /// Units are selected randomly to the flight phase.
    ///
    /// # Errors
    /// Returns an error if not all ids exist in balancing data.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::cube_method::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let s = SamplingOptions::new(p).unwrap().cube(&mut rng, m)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), CubeError>(())
    /// ```
    fn cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<ID>, CubeError>
    where
        BAL: PointSet<Id = ID, Value = f64>;
    /// Draw a sample using the cube method.
    /// The sample is balanced on the provided auxiliary variables in `balancing`.
    /// For fixed sized samples, the first auxiliary variable should be the probability vector.
    ///
    /// Units are selected in sequence to the flight phase.
    ///
    /// # Errors
    /// Returns an error if not all ids exist in balancing data.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::cube_method::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let s = SamplingOptions::new(p).unwrap().cube(&mut rng, m)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), CubeError>(())
    /// ```
    fn sequential_cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<ID>, CubeError>
    where
        BAL: PointSet<Id = ID, Value = f64>;
}

/// Provides spatially balanced CUBE sampling methods
pub trait LocalCubeSampling<ID, R>
where
    R: Rng,
{
    /// Draw a sample using the local cube method.
    /// The sample is balanced on the provided auxiliary variables in `balancing`, and spread in the
    /// space of the `spreading` auxiliaries.
    ///
    /// For fixed sized samples, the first auxiliary variable should be the probability vector.
    ///
    /// # Errors
    /// Returns an error if not all ids exist in balancing data.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::cube_method::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let bal = Matrix::new(vec![
    ///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
    ///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    /// ], 10).unwrap();
    /// let spr = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let s = SamplingOptions::new(p).unwrap()
    ///     .set_spreading(spr).unwrap()
    ///     .local_cube(&mut rng, bal)?;
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), CubeError>(())
    /// ```
    fn local_cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<ID>, CubeError>
    where
        BAL: PointSet<Id = ID, Value = f64>;
}
impl<R, PO, AUX> CubeSampling<PO::Id, R> for SamplingOptions<PO, AUX>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64>,
{
    #[inline]
    fn cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<PO::Id>, CubeError>
    where
        BAL: PointSet<Id = PO::Id, Value = f64>,
    {
        CubeMethod::new(self, balancing).map(|c| {
            c.sample(rng, RandomStrategy)
                .controller
                .to_sorted_sample_vec()
        })
    }
    #[inline]
    fn sequential_cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<PO::Id>, CubeError>
    where
        BAL: PointSet<Id = PO::Id, Value = f64>,
    {
        CubeMethod::new(self, balancing).map(|c| {
            c.sample(rng, SequentialStrategy)
                .controller
                .to_sorted_sample_vec()
        })
    }
}
impl<R, PO, P> LocalCubeSampling<PO::Id, R> for SamplingOptions<PO, SpreadingOptions<P>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id>,
{
    #[inline]
    fn local_cube<BAL>(&self, rng: &mut R, balancing: BAL) -> Result<Vec<PO::Id>, CubeError>
    where
        BAL: PointSet<Id = PO::Id, Value = f64>,
    {
        CubeMethod::new_spreading(self, balancing).map(|c| {
            c.sample(rng, SpatialStrategy)
                .controller
                .to_sorted_sample_vec()
        })
    }
}
