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

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;

use envisim_utils::kd_tree::searcher::KNearestNeighbourSearcher;
use envisim_utils::matrix::{
    Dimensions,
    Matrix,
    MatrixBase,
    MatrixDims,
    MatrixRef,
    RawData,
};
use envisim_utils::probabilities::FloatProbabilities;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::ProbabilitySpec;
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::spatial::{
    Number,
    PointSet,
};
use rustc_hash::FxSeededState;

use super::cube::{
    BasicCubeStrategy,
    CubeRunner,
    CubeStrategy,
    LocalCubeStrategy,
};
use crate::error::{
    SamplingError,
    SamplingResult,
};

pub struct CubeStratifiedRunner<'a, C, S, STRATA> {
    runner: CubeRunner<C, S>,
    strata: HashMap<STRATA, Vec<usize>, FxSeededState>,
    strata_vec: &'a [STRATA],
    org_probabilities: Cow<'a, [f64]>,
    balancing_data: MatrixRef<'a, f64>,
}
impl<'a, C, S, STRATA> CubeStratifiedRunner<'a, C, S, STRATA>
where
    C: SampleController<Store = FloatProbabilities>,
    S: CubeStrategy<C>,
    STRATA: Copy + Eq + Hash,
{
    fn new<PS, SOP, R, T>(
        options: &'a SamplingOptions<PS, SOP, MatrixBase<T>>,
        rng: &mut R,
        controller: C,
        strategy: S,
        strata_vec: &'a [STRATA],
    ) -> SamplingResult<Self>
    where
        PS: ProbabilitySpec,
        R: RandomNumberGenerator,
        T: RawData<Elem = f64>,
    {
        let org_probabilities = options.probabilities().as_f64_slice();
        let balancing_data = options.balancing()?.data().to_matrixref();

        let a_dims = MatrixDims::new(
            balancing_data.nrow(),
            balancing_data
                .ncol()
                .checked_add(1)
                .expect("b_dims to not overflow by adding 1"),
        );
        let adjusted_data = Matrix::from_value(0.0, a_dims);
        let c_dims = MatrixDims::new(
            a_dims.cols,
            a_dims
                .cols
                .checked_add(1)
                .expect("b_dims to not overflow by adding 2"),
        );
        let candidate_data = Matrix::from_value(0.0, c_dims);
        let runner = CubeRunner {
            controller,
            strategy,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data,
            candidate_data,
        };

        let strata = HashMap::<STRATA, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
            org_probabilities.len() / 10,
            FxSeededState::with_seed(rng.rusize()),
        );

        let this = Self {
            runner,
            strata,
            strata_vec,
            org_probabilities,
            balancing_data,
        };
        this.prepare()
    }
    fn prepare(mut self) -> SamplingResult<Self> {
        if self.strata_vec.len() != self.runner.controller.population_size() {
            return Err(SamplingError::IncorrectStratification);
        }

        let probabilities = self.org_probabilities.as_ref();

        for (i, prob) in probabilities.iter().copied().enumerate() {
            if !self.runner.controller.indices().contains(i) {
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
            self.runner.adjusted_data[(i, self.balancing_data.ncol().get())] = 1.0;
            for j in 0..self.balancing_data.ncol().get() {
                self.runner.adjusted_data[(i, j)] = self.balancing_data[(i, j)] / prob;
            }
        }

        Ok(self)
    }
    fn sample<R>(&mut self, rng: &mut R) -> Vec<usize>
    where
        R: RandomNumberGenerator,
    {
        self.flight_per_stratum(rng);
        if self.strata.is_empty() {
            return self.runner.controller.sample_mut().sort_to_vec();
        }
        self.flight_on_full(rng);
        if self.runner.controller.indices().is_empty() {
            return self.runner.controller.sample_mut().sort_to_vec();
        }
        self.landing_per_stratum(rng);
        self.runner.controller.sample_mut().sort_to_vec()
    }
    fn flight_per_stratum<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let mut removable_stratums = Vec::<STRATA>::new();
        for (stratum_key, stratum) in self.strata.iter_mut() {
            self.runner.strategy.reset_to_ids(
                &mut self.runner.controller,
                stratum,
                self.runner.adjusted_data.ncol().get() + 1,
            );

            self.runner.run_flight(rng);

            if self.runner.controller.indices().is_empty() {
                removable_stratums.push(*stratum_key);
                continue;
            }

            stratum.clear();
            stratum.extend_from_slice(self.runner.controller.indices().list());
        }

        for key in removable_stratums.iter() {
            self.strata.remove(key);
        }
    }
    fn flight_on_full<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let adj_data_dim = MatrixDims::new(
            self.balancing_data.nrow(),
            self.balancing_data
                .ncol()
                .checked_add(self.strata.len())
                .expect("to be able to add the number of strata to balancing"),
        );
        let cand_data_dim = MatrixDims::new(
            adj_data_dim.cols,
            adj_data_dim
                .cols
                .checked_add(1)
                .expect("to be able to add the number of strata + 1 to balancing"),
        );

        self.runner.adjusted_data.resize(adj_data_dim);
        self.runner.candidate_data.resize(cand_data_dim);

        let mut all_units = Vec::<usize>::new();

        for (si, (_, stratum)) in self.strata.iter().enumerate() {
            all_units.extend_from_slice(stratum);

            for &id in stratum.iter() {
                self.runner.adjusted_data[(id, si + self.balancing_data.ncol().get())] = 1.0;
            }
        }

        self.runner.strategy.reset_to_ids(
            &mut self.runner.controller,
            &mut all_units,
            self.runner.adjusted_data.ncol().get() + 1,
        );

        self.runner.run_flight(rng);

        // Fix stratas
        self.strata.clear();

        for &id in self.runner.controller.indices().list().iter() {
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
    fn landing_per_stratum<R>(&mut self, rng: &mut R)
    where
        R: RandomNumberGenerator,
    {
        let probabilities = self.org_probabilities.as_ref();

        let adj_data_dim = MatrixDims::new(
            self.balancing_data.nrow(),
            self.balancing_data
                .ncol()
                .checked_add(1)
                .expect("to be able to add 1 to balancing"),
        );
        let cand_data_dim = MatrixDims::new(
            adj_data_dim.cols,
            adj_data_dim
                .cols
                .checked_add(1)
                .expect("to be able to add 2 to balancing"),
        );

        self.runner.adjusted_data.resize(adj_data_dim);
        self.runner.candidate_data.resize(cand_data_dim);

        for (_key, stratum) in self.strata.iter_mut() {
            for &id in stratum.iter() {
                self.runner.adjusted_data[(id, 0)] = 1.0;
                for j in 0..self.balancing_data.ncol().get() {
                    self.runner.adjusted_data[(id, j + 1)] =
                        self.balancing_data[(id, j)] / probabilities[id];
                }
            }

            self.runner.strategy.reset_to_ids(
                &mut self.runner.controller,
                stratum,
                cand_data_dim.cols.get(),
            );

            self.runner.run_landing(rng);
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
/// # use envisim_samplr::*;
/// # use envisim_samplr::cube_method::*;
/// # use envisim_utils::random::*;
/// # use envisim_utils::matrix::*;
/// let mut rng = SmallRng::from_os_rng();
/// let bal_m = Matrix::new(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SamplingOptions::new_equal(10, 2)?.set_balancing(bal_m)?;
/// let s = cube_stratified(&mut rng, &options, &strata)?;
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
pub fn cube_stratified<R, PS, SOP, T, STRATA>(
    rng: &mut R,
    options: &SamplingOptions<PS, SOP, MatrixBase<T>>,
    strata: &[STRATA],
) -> SamplingResult<Vec<usize>>
where
    R: RandomNumberGenerator,
    PS: ProbabilitySpec,
    T: RawData<Elem = f64>,
    STRATA: Copy + Eq + Hash,
{
    let controller = options.to_controller_float();
    let strategy = BasicCubeStrategy();
    let mut runner = CubeStratifiedRunner::new(options, rng, controller, strategy, strata)?;
    Ok(runner.sample(rng))
}

/// Draw a sample using the stratified local cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// the sample is spatially balanced on the provided auxilliary variables in `auxiliaries`.
/// The first auxilliary variable should not be the probability vector.
/// For fixed sized samples, the probabilities in each strata must be integer.
///
/// # Examples
/// ```
/// # use envisim_samplr::*;
/// # use envisim_samplr::cube_method::*;
/// # use envisim_utils::random::*;
/// # use envisim_utils::matrix::Matrix;
/// let mut rng = SmallRng::from_os_rng();
/// let bal_m = Matrix::new(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10).unwrap();
/// let spr_m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SamplingOptions::new_equal(10, 2)?.set_balancing(bal_m)?.set_spreading(spr_m)?;
/// let s = local_cube_stratified(&mut rng, &options, &strata)?;
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
pub fn local_cube_stratified<R, PS, SOP, N, T, STRATA>(
    rng: &mut R,
    options: &SamplingOptions<PS, SOP, MatrixBase<T>>,
    strata: &[STRATA],
) -> SamplingResult<Vec<usize>>
where
    R: RandomNumberGenerator,
    PS: ProbabilitySpec,
    SOP: PointSet<N>,
    N: Number,
    T: RawData<Elem = f64>,
    STRATA: Copy + Eq + Hash,
{
    let controller = options.to_spreading_controller_float()?;
    let searcher = KNearestNeighbourSearcher::new(
        options
            .balancing()?
            .data()
            .ncol()
            .checked_add(1)
            .expect("to be able to add 1 to balancing"),
        controller.tree().data(),
    );
    let strategy = LocalCubeStrategy {
        spreading_options: options.spreading()?,
        searcher,
    };
    let mut runner = CubeStratifiedRunner::new(options, rng, controller, strategy, strata)?;
    Ok(runner.sample(rng))
}
