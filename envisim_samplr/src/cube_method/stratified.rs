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

//! Cube stratified methods

use std::num::NonZeroUsize;

use envisim_utils::matrix::{
    Dimensions,
    MatrixDims,
};
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::FloatRng;
use envisim_utils::sample_controller::TreeStorage;
use envisim_utils::sampling_options::{
    BalancingOptions,
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::{
    ConstructableDataView,
    DataView,
    PointSet,
};

use super::cube::{
    CubeMethod,
    CubeStrategy,
    RandomStrategy,
    SpatialStrategy,
};
use crate::error::{
    SamplingError,
    SamplingResult,
};

/// Stratified cube method runner
struct CubeStratifiedMethod<'bopts, PS, AUX, BL, PR, TR, SE, STRATA>
where
    PS: ProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
    STRATA: DataView<Id = PS::Id>,
{
    /// The main cube runner
    cube: CubeMethod<'bopts, PS, AUX, BL, PR, TR, SE>,
    /// The original vector of strata
    org_strata: STRATA,
    /// The stratification
    remaining_strata: Vec<STRATA::Value>,
}
impl<'bopts, PS, AUX, BL, PR, TR, SE, STRATA>
    CubeStratifiedMethod<'bopts, PS, AUX, BL, PR, TR, SE, STRATA>
where
    PS: ProbabilitiesSpec<Real = f64>,
    BL: PointSet<Id = PS::Id, Value = f64>,
    PR: ProbabilityStore<Id = PS::Id, N = f64>,
    TR: TreeStorage<PS::Id>,
    STRATA: DataView<Id = PS::Id, Value: Copy + Ord>,
{
    /// Constructs a new stratified runner
    /// # Panics
    /// Panics if balancing dims tend to overflow in small additions
    #[inline]
    fn new(
        mut cube: CubeMethod<'bopts, PS, AUX, BL, PR, TR, SE>,
        strata: STRATA,
    ) -> SamplingResult<CubeStratifiedMethod<'bopts, PS, AUX, BL, PR, TR, SE, STRATA>>
    where
        PS: ConstructableDataView,
    {
        let c_dims = MatrixDims::new(
            cube.cand_data.nrow(),
            cube.cand_data
                .ncol()
                .checked_add(1) // Add another for probabilities as this is not given by balopts
                .expect("b_dims to not overflow by adding 2"),
        );
        cube.cand_data.resize(c_dims);

        let mut remaining_strata =
            Vec::<STRATA::Value>::with_capacity(strata.len().saturating_div(10));
        for (id, stratum) in strata.entries() {
            // Check strata IDs
            if !cube
                .controller
                .probabilities()
                .is_partial(id)
                .ok_or(SamplingError::IncorrectStratification)?
            {
                // Skip any non-partial.
                continue;
            }

            if let Err(idx) = remaining_strata.binary_search(stratum) {
                // Only add stratum that hasnt been added already
                remaining_strata.insert(idx, *stratum);
            }
        }

        Ok(CubeStratifiedMethod {
            cube,
            org_strata: strata,
            remaining_strata,
        })
    }
    /// Fly each stratum
    #[inline]
    fn flight_per_stratum<R, S>(&mut self, rng: &mut R, strategy: S)
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        let b_cols = self.cube.options.balancing().data().dimensions();
        let c_cols = b_cols.checked_add(1).expect("no overflow"); // Add one for probs
        let n_candidates = c_cols.checked_add(1).expect("no overflow"); // Select one more than cols

        self.cube.cand_data.resize((c_cols, n_candidates));

        let mut id_vec = Vec::<PS::Id>::with_capacity(self.cube.controller.population_size().get());
        let mut stratum_i = 0;

        while stratum_i < self.remaining_strata.len() {
            let stratum = self.remaining_strata[stratum_i];
            // Reset ids to current stratum

            id_vec.clear();
            id_vec.extend(
                self.org_strata
                    .entries()
                    .filter(|(id, st)| {
                        // Filter out current stratum
                        stratum == **st
                        // Filter out resolved
                            && self
                                .cube
                                .controller
                                .probabilities()
                                .is_partial(*id)
                                .expect("id exists")
                    })
                    .map(|(id, _)| id),
            );
            strategy.reset_to_ids(&mut self.cube, &mut id_vec);

            // Run flight
            while self.cube.controller.indices().len() >= n_candidates.get() {
                strategy.select_units(rng, &mut self.cube, n_candidates);

                assert_eq!(
                    self.cube.candidates.len(),
                    n_candidates.get(),
                    "incorrect number of candidates selected"
                );

                // Set candidate data
                for (ci, &id) in self.cube.candidates.iter().enumerate() {
                    let p = self
                        .cube
                        .options
                        .probabilities()
                        .get_real(id)
                        .expect("id to exist in probabilities");

                    // In flight, all balancing will be respected, so p can be last col here
                    self.cube.cand_data[(b_cols.get(), ci)] = 1.0;

                    // Set balancing data
                    for (j, v) in self
                        .cube
                        .options
                        .balancing()
                        .data()
                        .coords(id)
                        .expect("id to exist")
                        .enumerate()
                    {
                        self.cube.cand_data[(j, ci)] = *v / p;
                    }
                }

                self.cube.update_probabilities(rng);
            }

            // Maybe remove statum?
            if self.cube.controller.indices().is_empty() {
                self.remaining_strata.swap_remove(stratum_i);
            } else {
                stratum_i += 1;
            }
        }
    }
    /// Runs the flight phase for all remaining units
    #[inline]
    fn flight_on_full<R, S>(&mut self, rng: &mut R, strategy: S)
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        let b_cols = self.cube.options.balancing().data().dimensions();
        let c_cols = b_cols
            .checked_add(self.remaining_strata.len()) // Add for each remain. strata
            .expect("no overflow");
        let n_candidates = c_cols.checked_add(1).expect("no overflow");

        // resize cand data
        self.cube.cand_data.resize((c_cols, n_candidates));

        let mut id_vec: Vec<PS::Id> = self
            .org_strata
            .ids()
            // Filter out resolved
            .filter(|id| {
                self.cube
                    .controller
                    .probabilities()
                    .is_partial(*id)
                    .expect("id to exist")
            })
            .collect();
        strategy.reset_to_ids(&mut self.cube, &mut id_vec);

        // Run flight
        while self.cube.controller.indices().len() >= n_candidates.get() {
            strategy.select_units(rng, &mut self.cube, n_candidates);

            assert_eq!(
                self.cube.candidates.len(),
                n_candidates.get(),
                "incorrect number of candidates selected"
            );

            // Set candidate data
            for (i, &id) in self.cube.candidates.iter().enumerate() {
                let p = self
                    .cube
                    .options
                    .probabilities()
                    .get_real(id)
                    .expect("id to exist in probabilities");
                let st = *self.org_strata.get(id).expect("id to exist");

                // Set balancing data
                for (j, v) in self
                    .cube
                    .options
                    .balancing()
                    .data()
                    .coords(id)
                    .expect("id to exist")
                    .enumerate()
                {
                    self.cube.cand_data[(j, i)] = *v / p;
                }

                // In flight, all balancing will be respected, so p's can be last cols here
                for (j, &stratum) in self.remaining_strata.iter().enumerate() {
                    self.cube.cand_data[(b_cols.get() + j, i)] =
                        if stratum == st { 1.0 } else { 0.0 };
                }
            }

            self.cube.update_probabilities(rng);
        }
    }
    /// Runs the landing phase for each stratum
    #[inline]
    fn landing_per_stratum<R, S>(&mut self, rng: &mut R, strategy: S)
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        let b_cols = self.cube.options.balancing().data().dimensions();
        let c_cols = b_cols.checked_add(1).expect("no overflow"); // Add one for probs

        let mut id_vec = Vec::<PS::Id>::with_capacity(self.cube.controller.population_size().get());

        for &stratum in &self.remaining_strata {
            id_vec.clear();
            id_vec.extend(
                self.org_strata
                    .entries()
                    .filter(|(id, st)| {
                        // Filter out current stratum
                        stratum == **st
                        // Filter out resolved
                            && self
                                .cube
                                .controller
                                .probabilities()
                                .is_partial(*id)
                                .expect("id exists")
                    })
                    .map(|(id, _)| id),
            );
            strategy.reset_to_ids(&mut self.cube, &mut id_vec);

            // Run landing only if we have lass than c_cols units left in a statum
            assert!(
                self.cube.controller.indices().len() <= c_cols.get(),
                "landing phase committed early ({} remaining, {} max)",
                self.cube.controller.indices().len(),
                b_cols.get()
            );

            while self.cube.controller.indices().len() > 1 {
                // Candidates consists of all remaining units
                let n_candidates =
                    NonZeroUsize::new(self.cube.controller.indices().len()).expect(">1");
                self.cube.set_candidates_sequentially(NonZeroUsize::MAX);
                assert_eq!(
                    n_candidates.get(),
                    self.cube.candidates.len(),
                    "incorrect number of candidates"
                );
                let c_cols_adj = NonZeroUsize::new(n_candidates.get() - 1).expect(">0");

                // resize cand data
                self.cube.cand_data.resize((c_cols_adj, n_candidates));

                // Set candidate data
                for (i, &id) in self.cube.candidates.iter().enumerate() {
                    let p = self
                        .cube
                        .options
                        .probabilities()
                        .get_real(id)
                        .expect("id to exist in probabilities");

                    // In landing, probs must be first so it def. get respected
                    self.cube.cand_data[(0, i)] = 1.0;

                    // Set balancing data
                    for (j, v) in self
                        .cube
                        .options
                        .balancing()
                        .data()
                        .coords(id)
                        .expect("id to exist")
                        .enumerate()
                        .take(c_cols_adj.get() - 1)
                    {
                        self.cube.cand_data[(j + 1, i)] = *v / p;
                    }
                }

                self.cube.update_probabilities(rng);
            }

            self.cube.controller.unit_decide_last(rng);
        }
    }
    /// Runs the algorithm and returns the sample vector
    #[must_use]
    #[inline]
    fn sample<R, S>(mut self, rng: &mut R, strategy: S) -> Vec<PS::Id>
    where
        R: FloatRng,
        S: CubeStrategy<'bopts, PS, AUX, BL, PR, Tree = TR, Searcher = SE>,
    {
        self.flight_per_stratum(rng, strategy);
        if self.remaining_strata.is_empty() {
            return self.cube.controller.to_sorted_sample_vec();
        }
        self.flight_on_full(rng, strategy);
        if self.cube.controller.indices().is_empty() {
            return self.cube.controller.to_sorted_sample_vec();
        }
        self.landing_per_stratum(rng, strategy);
        self.cube.controller.to_sorted_sample_vec()
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
/// let mut rng = try_sys_rng().unwrap();
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
/// # Errors
/// Returns an error if the length of `strata` does not match the population size.
#[inline]
pub fn cube_stratified<R, PO, AUX, BL, STRATA>(
    rng: &mut R,
    options: &SamplingOptions<PO, AUX, BalancingOptions<BL>>,
    strata: STRATA,
) -> SamplingResult<Vec<PO::Id>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64> + ConstructableDataView,
    BL: PointSet<Id = PO::Id, Value = f64>,
    STRATA: DataView<Id = PO::Id, Value: Copy + Ord>,
{
    let cube = CubeMethod::new(options);
    Ok(CubeStratifiedMethod::new(cube, strata)?.sample(rng, RandomStrategy))
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
/// # use envisim_utils::matrix::*;
/// let mut rng = try_sys_rng().unwrap();
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
/// # Errors
/// Returns an error if the length of `strata` does not match the population size.
#[inline]
pub fn local_cube_stratified<R, PO, P, BL, STRATA>(
    rng: &mut R,
    options: &SamplingOptions<PO, SpreadingOptions<P>, BalancingOptions<BL>>,
    strata: STRATA,
) -> SamplingResult<Vec<PO::Id>>
where
    R: SamplingOptionsRng<PO>,
    PO: ProbabilitiesSpec<Real = f64> + ConstructableDataView,
    P: PointSet<Id = PO::Id>,
    BL: PointSet<Id = PO::Id, Value = f64>,
    STRATA: DataView<Id = PO::Id, Value: Copy + Ord>,
{
    let cube = CubeMethod::new_spreading(options);
    Ok(CubeStratifiedMethod::new(cube, strata)?.sample(rng, SpatialStrategy))
}
