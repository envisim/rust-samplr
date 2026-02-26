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

//! Cube method designs

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroUsize;

use envisim_utils::kd_tree::Searcher;
use envisim_utils::matrix::{
    Matrix,
    MatrixIndex,
};
use envisim_utils::probabilities::{
    Probabilities,
    ProbabilitiesUnequal,
};
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    BalancingOptions,
    Enabled,
    SpreadingOptions,
};
pub use envisim_utils::sampling_options::{
    SamplingOptions,
    SamplingOptionsError,
};
use rustc_hash::FxSeededState;

pub use crate::SamplingError;
use crate::sample_controller::{
    BasicSampleController,
    SampleController,
    SpreadingSampleController,
};
use crate::srs::srs;

pub struct BaseCube<'a, C>
where
    C: SampleController,
{
    controller: C,
    candidates: Vec<usize>,
    adjusted_data: Matrix<'a>,
    candidate_data: Matrix<'a>,
}
impl<'a, C> BaseCube<'a, C>
where
    C: SampleController<Probs = ProbabilitiesUnequal>,
{
    fn new<P, S>(options: &'a SamplingOptions<'a, P, S, Enabled>, controller: C) -> Self
    where
        P: Probabilities,
    {
        // let balancing_data = container.options().check_balancing()?.balancing().unwrap();
        let balancing_data = options.balancing().data();
        let b_dims = balancing_data.dims();
        let mut adjusted_data = Matrix::new(balancing_data.data(), b_dims.row())
            .expect("balancing data should be non-empty");

        for i in 0..b_dims.row() {
            let p = controller.probabilities().get(i);
            for j in 0..b_dims.col() {
                adjusted_data[(i, j)] /= p;
            }
        }

        Self {
            controller,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data,
            candidate_data: Matrix::from_value(0.0, (b_dims.col(), b_dims.col() + 1)).unwrap(),
        }
    }
    fn set_candidate_data(&mut self) {
        let n_candidates = self.candidates.len();
        assert!(n_candidates <= self.adjusted_data.ncol() + 1);
        let dims = MatrixIndex(n_candidates - 1, n_candidates);
        self.candidate_data.resize(dims);

        for (i, &id) in self.candidates.iter().enumerate() {
            for j in 0..dims.row() {
                self.candidate_data[(j, i)] = self.adjusted_data[(id, j)];
            }
        }
    }
    fn clear_candidates(&mut self) { self.candidates.clear(); }
    fn set_candidates_from_indices(&mut self, len: usize) {
        let number_of_remaining_units = self.controller.indices().len();
        let len = if len == 0 || len > number_of_remaining_units {
            number_of_remaining_units
        } else {
            len
        };
        assert!(len <= self.adjusted_data.ncol() + 1);

        // Set candidates
        self.candidates.clear();
        self.candidates
            .extend_from_slice(&self.controller.indices().list()[0..len]);

        // Set data
        self.set_candidate_data();
    }
    fn update_probabilities<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
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
impl<'a, P, S> From<&'a SamplingOptions<'a, P, S, Enabled>>
    for BaseCube<'a, BasicSampleController<ProbabilitiesUnequal>>
where
    P: Probabilities,
    BasicSampleController<ProbabilitiesUnequal>: From<&'a SamplingOptions<'a, P, S, Enabled>>,
{
    fn from(options: &'a SamplingOptions<'a, P, S, Enabled>) -> Self {
        let controller = options.into();
        BaseCube::new(options, controller)
    }
}
impl<'a, P> From<&'a SamplingOptions<'a, P, Enabled, Enabled>>
    for BaseCube<'a, SpreadingSampleController<'a, ProbabilitiesUnequal>>
where
    P: Probabilities,
    SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, Enabled>>,
{
    fn from(options: &'a SamplingOptions<'a, P, Enabled, Enabled>) -> Self {
        let controller = options.into();
        BaseCube::new(options, controller)
    }
}

pub trait CubeMethod<'a> {
    type Controller: SampleController<Probs = ProbabilitiesUnequal>;
    fn base(&self) -> &BaseCube<'a, Self::Controller>;
    fn base_mut(&mut self) -> &mut BaseCube<'a, Self::Controller>;
    // fn controller(&self) -> &C { &self.base().controller }
    // fn controller_mut(&mut self) -> &mut C { &mut self.base_mut().controller }
    fn sample<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Vec<usize> {
        self.run(rng);
        self.base_mut().controller.sample_mut().sort_to_vec()
    }
    fn run<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        self.run_flight(rng);
        self.run_landing(rng);
    }
    fn run_flight<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let b_cols = self.base().adjusted_data.ncol();
        assert_eq!(b_cols, self.base().candidate_data.nrow());

        while self.base().controller.indices().len() > b_cols {
            self.select_units(rng, b_cols + 1);
            self.base_mut().set_candidate_data();
            self.base_mut().update_probabilities(rng);
        }
    }
    fn run_landing<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let b_cols = self.base().adjusted_data.ncol();
        let len = self.base().controller.indices().len();
        assert!(
            len <= b_cols,
            "landing phase committed early: {len} units remaining, with {b_cols} cols",
        );

        while self.base().controller.indices().len() > 1 {
            self.base_mut().set_candidates_from_indices(0);
            self.base_mut().update_probabilities(rng)
        }

        self.base_mut()
            .controller
            .unit_decide_last(rng)
            .expect("last unit to be decided");
    }

    fn select_units<R: RandomNumberGenerator>(&mut self, rng: &mut R, n_units: usize);
    // Used for stratified
    fn reset_to_ids(&mut self, ids: &mut [usize], n_neighbours: usize);
}

struct Cube<'a> {
    base: BaseCube<'a, BasicSampleController<ProbabilitiesUnequal>>,
}
impl<'a> CubeMethod<'a> for Cube<'a> {
    type Controller = BasicSampleController<ProbabilitiesUnequal>;
    fn base(&self) -> &BaseCube<'a, BasicSampleController<ProbabilitiesUnequal>> { &self.base }
    fn base_mut(&mut self) -> &mut BaseCube<'a, BasicSampleController<ProbabilitiesUnequal>> {
        &mut self.base
    }
    fn select_units<R: RandomNumberGenerator>(&mut self, _: &mut R, n_units: usize) {
        self.base.set_candidates_from_indices(n_units);
    }
    fn reset_to_ids(&mut self, ids: &mut [usize], _n_neighbours: usize) {
        self.base.controller.indices_mut().clear();
        for &id in ids.iter() {
            self.base.controller.indices_mut().insert(id).unwrap();
        }
    }
}
impl<'a> Cube<'a> {
    pub fn new<P, S>(options: &'a SamplingOptions<'a, P, S, Enabled>) -> Self
    where
        P: Probabilities,
        BasicSampleController<ProbabilitiesUnequal>: From<&'a SamplingOptions<'a, P, S, Enabled>>,
    {
        Self {
            base: options.into(),
        }
    }
}
/// Draw a sample using the cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// For fixed sized samples, the first auxilliary variable should be the probability vector.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::random::*;
/// use envisim_utils::matrix::Matrix;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let opts = SamplingOptions::new(&p)?.set_balancing(&bal_m)?;
/// let s= cube(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tillé, Y. (2004).
/// Efficient balanced sampling: the cube method.
/// Biometrika, 91(4), 893-912.
/// <https://doi.org/10.1093/biomet/91.4.893>
pub fn cube<R, P, S>(rng: &mut R, options: &SamplingOptions<'_, P, S, Enabled>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, S, Enabled>>,
{
    Cube::new(options).sample(rng)
}

struct LocalCube<'a> {
    base: BaseCube<'a, SpreadingSampleController<'a, ProbabilitiesUnequal>>,
    spreading_options: &'a SpreadingOptions<'a>,
    searcher: Searcher,
}
impl<'a> CubeMethod<'a> for LocalCube<'a> {
    type Controller = SpreadingSampleController<'a, ProbabilitiesUnequal>;
    fn base(&self) -> &BaseCube<'a, SpreadingSampleController<'a, ProbabilitiesUnequal>> {
        &self.base
    }
    fn base_mut(
        &mut self,
    ) -> &mut BaseCube<'a, SpreadingSampleController<'a, ProbabilitiesUnequal>> {
        &mut self.base
    }
    fn select_units<R: RandomNumberGenerator>(&mut self, rng: &mut R, n_units: usize) {
        assert!(n_units > 1);
        let len = self.base.controller.indices().len();
        assert!(len >= n_units);

        if len == n_units {
            self.base.set_candidates_from_indices(n_units);
            return;
        }

        self.base.clear_candidates();

        // Draw the first unit at random
        let id1 = self.base.controller.indices().draw(rng).unwrap();
        self.base.candidates.push(id1);

        // Find the neighbours of this first unit
        self.searcher
            .find_neighbours_of_id(self.base.controller.tree(), id1)
            .unwrap();

        // Add all neighbours, if no equals
        if self.searcher.neighbours().len() == n_units - 1 {
            self.base
                .candidates
                .extend_from_slice(self.searcher.neighbours());
            return;
        }

        // There exists multiple max_distance neighbours, we need to add the non max, and then
        // sample amongst the max'es
        let mut i: usize = 0;
        let maximum_distance = self
            .searcher
            .distance_k(self.searcher.neighbours().len() - 1);

        // Add all neighbours that are not on maximum distance
        while i < n_units - 1 && self.searcher.distance_k(i) < maximum_distance {
            let id = self.searcher.neighbours()[i];
            self.base.candidates.push(id);
            i += 1;
        }

        // Randomly add neighbours on the maximum distance
        // We need to draw from the
        let n_remaining_units = self.searcher.neighbours().len() - i;
        // the number left to fill amongst the candidates
        let n_open_spots = n_units - self.base.candidates.len();
        let opts = SamplingOptions::new_equal(n_remaining_units, n_open_spots).unwrap();

        let s = srs(rng, &opts);
        for k in s {
            self.base.candidates.push(self.searcher.neighbours()[i + k]);
        }
    }
    fn reset_to_ids(&mut self, ids: &mut [usize], n_neighbours: usize) {
        self.searcher.set_n_neighbours(
            NonZeroUsize::new(n_neighbours).expect("n_neighbours to be positive"),
        );

        self.base.controller.indices_mut().clear();
        self.base
            .controller
            .reset_tree(self.spreading_options, ids)
            .expect("tree should be resettable");

        for id in ids.iter() {
            self.base.controller.indices_mut().insert(*id).unwrap();
        }
    }
}
impl<'a> LocalCube<'a> {
    pub fn new<P>(options: &'a SamplingOptions<'a, P, Enabled, Enabled>) -> Self
    where
        P: Probabilities,
        SpreadingSampleController<'a, ProbabilitiesUnequal>:
            From<&'a SamplingOptions<'a, P, Enabled, Enabled>>,
    {
        let base: BaseCube<SpreadingSampleController<ProbabilitiesUnequal>> = options.into();
        let cols = options.balancing().data().ncol();
        let searcher = Searcher::new(
            base.controller.tree(),
            NonZeroUsize::new(cols).expect("balancing to have columns"),
        );
        Self {
            base,
            spreading_options: options.spreading(),
            searcher,
        }
    }
}
/// Draw a sample using the local cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// the sample is spatially balanced on the provided auxilliary variables in `auxiliaries`.
/// For fixed sized samples, the first auxilliary variable should be the probability vector.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::random::*;
/// use envisim_utils::matrix::Matrix;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let spr_m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10)
///     .unwrap();
/// let opts = SamplingOptions::new(&p)?
///     .set_balancing(&bal_m)?
///     .set_spreading(&spr_m)?;
/// let s = local_cube(&mut rng, &opts);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingOptionsError>(())
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
pub fn local_cube<R, P>(
    rng: &mut R,
    options: &SamplingOptions<'_, P, Enabled, Enabled>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, Enabled>>,
{
    LocalCube::new(options).sample(rng)
}

pub struct CubeStratified<'a, T>
where
    T: CubeMethod<'a>,
{
    cube: T,
    strata: HashMap<i64, Vec<usize>, FxSeededState>,
    strata_vec: &'a [i64],
    org_probabilities: Cow<'a, [f64]>,
    balancing_options: &'a BalancingOptions<'a>,
}
impl<'a, T> CubeStratified<'a, T>
where
    T: CubeMethod<'a>,
{
    fn prepare(&mut self) -> Result<&mut Self, SamplingError> {
        if self.strata_vec.len() != self.cube.base().controller.population_size() {
            return Err(SamplingError::IncorrectStratification);
        }

        let balancing_data = self.balancing_options.data();
        let probabilities = self.org_probabilities.as_ref();
        let population_size = self.org_probabilities.len();

        for i in 0..population_size {
            if !self.cube.base().controller.indices().contains(i) {
                continue;
            }

            let stratum = self.strata_vec[i];
            match self.strata.get_mut(&stratum) {
                Some(uvec) => {
                    uvec.push(i);
                }
                None => {
                    self.strata.insert(stratum, vec![i]);
                }
            };

            // Order doesn't matter during flight
            self.cube.base_mut().adjusted_data[(i, balancing_data.ncol())] = 1.0;
            for j in 0..balancing_data.ncol() {
                self.cube.base_mut().adjusted_data[(i, j)] =
                    balancing_data[(i, j)] / probabilities[i];
            }
        }

        Ok(self)
    }
    fn sample<R: RandomNumberGenerator>(&mut self, rng: &mut R) -> Vec<usize> {
        self.flight_per_stratum(rng);
        if self.strata.is_empty() {
            return self.cube.base_mut().controller.sample_mut().sort_to_vec();
        }
        self.flight_on_full(rng);
        if self.cube.base().controller.indices().is_empty() {
            return self.cube.base_mut().controller.sample_mut().sort_to_vec();
        }
        self.landing_per_stratum(rng);
        self.cube.base_mut().controller.sample_mut().sort_to_vec()
    }
    fn flight_per_stratum<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let mut removable_stratums = Vec::<i64>::new();
        for (stratum_key, stratum) in self.strata.iter_mut() {
            self.cube
                .reset_to_ids(stratum, self.cube.base().adjusted_data.ncol() + 1);

            self.cube.run_flight(rng);

            if self.cube.base().controller.indices().is_empty() {
                removable_stratums.push(*stratum_key);
                continue;
            }

            stratum.clear();
            stratum.extend_from_slice(self.cube.base().controller.indices().list());
        }

        for key in removable_stratums.iter() {
            self.strata.remove(key);
        }
    }
    fn flight_on_full<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let balancing_data = self.balancing_options.data();

        let adj_data_dim = MatrixIndex(
            balancing_data.nrow(),
            balancing_data.ncol() + self.strata.len(),
        );
        let cand_data_dim = MatrixIndex(adj_data_dim.col(), adj_data_dim.col() + 1);

        self.cube.base_mut().adjusted_data.resize(adj_data_dim);
        self.cube.base_mut().candidate_data.resize(cand_data_dim);

        let mut all_units = Vec::<usize>::new();

        for (si, (_, stratum)) in self.strata.iter().enumerate() {
            all_units.extend_from_slice(stratum);

            for &id in stratum.iter() {
                self.cube.base_mut().adjusted_data[(id, si + balancing_data.ncol())] = 1.0;
            }
        }

        self.cube
            .reset_to_ids(&mut all_units, self.cube.base().adjusted_data.ncol() + 1);

        self.cube.run_flight(rng);

        // Fix stratas
        self.strata.clear();

        for &id in self.cube.base().controller.indices().list().iter() {
            let stratum = self.strata_vec[id];
            match self.strata.get_mut(&stratum) {
                Some(uvec) => {
                    uvec.push(id);
                }
                None => {
                    self.strata.insert(stratum, vec![id]);
                }
            };
        }
    }
    fn landing_per_stratum<R: RandomNumberGenerator>(&mut self, rng: &mut R) {
        let balancing_data = self.balancing_options.data();
        let probabilities = self.org_probabilities.as_ref();

        let adj_data_dim = MatrixIndex(balancing_data.nrow(), balancing_data.ncol() + 1);
        let cand_data_dim = MatrixIndex(adj_data_dim.col(), adj_data_dim.col() + 1);

        self.cube.base_mut().adjusted_data.resize(adj_data_dim);
        self.cube.base_mut().candidate_data.resize(cand_data_dim);

        for (_key, stratum) in self.strata.iter_mut() {
            for &id in stratum.iter() {
                self.cube.base_mut().adjusted_data[(id, 0)] = 1.0;
                for j in 0..balancing_data.ncol() {
                    self.cube.base_mut().adjusted_data[(id, j + 1)] =
                        balancing_data[(id, j)] / probabilities[id];
                }
            }

            self.cube.reset_to_ids(stratum, cand_data_dim.col());

            self.cube.run_landing(rng);
        }
    }
}

/// Draw a sample using the stratified cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// The first auxilliary variable should not be the probability vector.
/// For fixed sized samples, the probabilities in each strata must be integer.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::random::*;
/// use envisim_utils::matrix::Matrix;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2; 10];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SamplingOptions::new(&p)?.set_balancing(&bal_m)?;
/// let s = cube_stratified(&mut rng, &options, &strata)?;
///
/// assert_eq!(s.len(), 2);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Chauvet, G. (2009).
/// Stratified balanced sampling.
/// Survey Methodology, 35(1), 115-119.
///
/// Deville, J. C., & Tillé, Y. (2004).
/// Efficient balanced sampling: the cube method.
/// Biometrika, 91(4), 893-912.
/// <https://doi.org/10.1093/biomet/91.4.893>
pub fn cube_stratified<R, P, S>(
    rng: &mut R,
    options: &SamplingOptions<'_, P, S, Enabled>,
    strata: &[i64],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, S, Enabled>>,
{
    let balancing_data = options.balancing();
    let org_probabilities = options.probabilities().slice();

    let controller: BasicSampleController<ProbabilitiesUnequal> = options.into();
    let seed = rng.rusize();

    let mut cs = CubeStratified {
        cube: Cube {
            base: BaseCube::<BasicSampleController<ProbabilitiesUnequal>> {
                controller,
                candidates: Vec::<usize>::with_capacity(20),
                adjusted_data: Matrix::from_value(
                    0.0,
                    (
                        balancing_data.data().nrow(),
                        balancing_data.data().ncol() + 1,
                    ),
                )
                .unwrap(),
                candidate_data: Matrix::from_value(
                    0.0,
                    (
                        balancing_data.data().ncol() + 1,
                        balancing_data.data().ncol() + 2,
                    ),
                )
                .unwrap(),
            },
        },
        strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
            org_probabilities.len() / 10,
            FxSeededState::with_seed(seed),
        ),
        strata_vec: strata,
        org_probabilities,
        balancing_options: balancing_data,
    };

    cs.prepare().map(|s| s.sample(rng))
}

/// Draw a sample using the stratified local cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// the sample is spatially balanced on the provided auxilliary variables in `auxiliaries`.
/// The first auxilliary variable should not be the probability vector.
/// For fixed sized samples, the probabilities in each strata must be integer.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::random::*;
/// use envisim_utils::matrix::Matrix;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2; 10];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let spr_m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10)
///     .unwrap();
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SamplingOptions::new(&p)?.set_balancing(&bal_m)?.set_spreading(&spr_m)?;
/// let s = local_cube_stratified(&mut rng, &options, &strata)?;
///
/// assert_eq!(s.len(), 2);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Chauvet, G. (2009).
/// Stratified balanced sampling.
/// Survey Methodology, 35(1), 115-119.
///
/// Deville, J. C., & Tillé, Y. (2004).
/// Efficient balanced sampling: the cube method.
/// Biometrika, 91(4), 893-912.
/// <https://doi.org/10.1093/biomet/91.4.893>
///
/// Grafström, A., & Tillé, Y. (2013).
/// Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
/// Environmetrics, 24(2), 120-131.
/// <https://doi.org/10.1002/env.2194>
pub fn local_cube_stratified<R, P>(
    rng: &mut R,
    options: &SamplingOptions<'_, P, Enabled, Enabled>,
    strata: &[i64],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> SpreadingSampleController<'a, ProbabilitiesUnequal>:
        From<&'a SamplingOptions<'a, P, Enabled, Enabled>>,
{
    let balancing_data = options.balancing();
    let org_probabilities = options.probabilities().slice();

    let controller: SpreadingSampleController<ProbabilitiesUnequal> = options.into();
    let searcher = Searcher::new(
        controller.tree(),
        NonZeroUsize::new(balancing_data.data().ncol() + 1).unwrap(),
    );
    let seed = rng.rusize();

    let mut cs = CubeStratified {
        cube: LocalCube {
            base: BaseCube::<SpreadingSampleController<ProbabilitiesUnequal>> {
                controller,
                candidates: Vec::<usize>::with_capacity(20),
                adjusted_data: Matrix::from_value(
                    0.0,
                    (
                        balancing_data.data().nrow(),
                        balancing_data.data().ncol() + 1,
                    ),
                )
                .unwrap(),
                candidate_data: Matrix::from_value(
                    0.0,
                    (
                        balancing_data.data().ncol() + 1,
                        balancing_data.data().ncol() + 2,
                    ),
                )
                .unwrap(),
            },
            spreading_options: options.spreading(),
            searcher,
        },
        strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
            org_probabilities.len() / 10,
            FxSeededState::with_seed(seed),
        ),
        strata_vec: strata,
        org_probabilities,
        balancing_options: balancing_data,
    };

    cs.prepare().map(|s| s.sample(rng))
}

/// Finds a vector in null space of a (n-1)*n matrix. The matrix is mutated into rref.
fn find_vector_in_null_space(mat: &mut Matrix) -> Vec<f64> {
    let MatrixIndex(nrow, ncol) = mat.dims();
    assert!(nrow > 0);
    assert!(nrow == ncol - 1);

    mat.reduced_row_echelon_form();
    // If (0, 0) == 0.0, then the we have big problems
    assert!(mat[(0, 0)] != 0.0);

    let mut v = vec![1.0; ncol];

    // If (n-1, n-1) = 1.0, then we have linearly independent rows,
    // and the form of the matrix is an identity matrix with the parts
    // of the null space vector in the last column
    if mat[(nrow - 1, nrow - 1)] == 1.0 {
        for i in 0..nrow {
            v[i] = -mat[(i, ncol - 1)];
        }

        return v;
    }

    let mut pivot_cols = Vec::with_capacity(nrow);
    let mut is_pivot = vec![false; ncol];

    for row in 0..nrow {
        for col in 0..ncol {
            if mat[(row, col)] != 0.0 {
                // Found first non-zero entry in row

                if mat[(row, col)] == 1.0 {
                    pivot_cols.push(col);
                    is_pivot[col] = true;
                }
                break;
            }
        }
    }

    // Build null space vector
    // Free variables (non-pivot columns) alternating set to +/- 1
    // Basic vars (pivot cols) computet to satisfy Ax = 0

    // Set free variables
    let mut free_idx = 0;
    for col in 0..ncol {
        if !is_pivot[col] {
            v[col] = if free_idx % 2 == 0 { 1.0 } else { -1.0 };
            free_idx += 1;
        }
    }

    // Compute basic variables working backwards through rows
    for (row, &pivot_col) in pivot_cols.iter().enumerate().rev() {
        let mut sum = 0.0;
        for col in (pivot_col + 1)..ncol {
            sum += mat[(row, col)] * v[col];
        }
        v[pivot_col] = -sum;
    }

    v
}

#[cfg(test)]
mod tests {
    use envisim_test_utils::*;

    use super::*;

    #[test]
    fn null() {
        let mut mat1 = Matrix::from_vec(
            vec![
                1.0, 2.0, 3.0, 1.0, //
                5.0, 10.0, 1.0, 5.0, //
                10.0, 1.0, 5.0, 10.0, //
            ],
            3,
        )
        .unwrap();
        mat1.reduced_row_echelon_form();
        assert!(
            mat1.data()
                == [
                    1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
                ]
        );
        let mat1_nullvec = find_vector_in_null_space(&mut mat1);
        assert_fvec(&mat1.prod_vec(&mat1_nullvec).unwrap(), &[0.0, 0.0, 0.0]);

        let mut mat2 = Matrix::from_vec(
            vec![
                1.0, 2.0, 3.0, 1.0, //
                5.0, 10.0, 10.0, 5.0, //
                1.0, 1.0, 5.0, 11.0, //
            ],
            3,
        )
        .unwrap();
        mat2.reduced_row_echelon_form();
        assert!(&mat2.data()[0..9] == vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_fvec(
            &mat2.data()[9..12], // col 3
            &[-2.5, 1.833333333333333, 0.166666666666667],
        );
        let mat2_nullvec = find_vector_in_null_space(&mut mat2);
        assert_fvec(&mat2.prod_vec(&mat2_nullvec).unwrap(), &[0.0, 0.0, 0.0]);
    }
}
