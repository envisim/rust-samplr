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

use crate::srs;
use crate::utils::SampleContainer;
pub use crate::{SampleOptions, SamplingError};
use envisim_utils::{kd_tree::Searcher, random::RandomNumberGenerator, InputError, Matrix};
use rustc_hash::FxSeededState;
use std::collections::HashMap;
use std::num::NonZeroUsize;

pub struct VariantCube {}
pub struct VariantLocalCube {
    searcher: Searcher,
}

pub struct CubeMethod<'a, R, T>
where
    R: RandomNumberGenerator + ?Sized,
    T: CubeMethodVariant<'a, R>,
{
    container: SampleContainer<'a, R>,
    variant: T,
    candidates: Vec<usize>,
    adjusted_data: Matrix<'a>,
    candidate_data: Matrix<'a>,
}

pub trait CubeMethodVariant<'a, R>
where
    R: RandomNumberGenerator + ?Sized,
{
    fn new(
        rng: &'a mut R,
        options: &'a SampleOptions<'a>,
    ) -> Result<CubeMethod<'a, R, Self>, SamplingError>
    where
        Self: Sized;
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &mut SampleContainer<'a, R>,
        n_units: usize,
    );
}

impl<'a, R, T> CubeMethod<'a, R, T>
where
    R: RandomNumberGenerator + ?Sized,
    T: CubeMethodVariant<'a, R>,
{
    #[inline]
    fn new(container: SampleContainer<'a, R>, variant: T) -> Result<Self, SamplingError> {
        let balancing_data = container.options().check_balancing()?.balancing().unwrap();
        let (b_nrow, b_ncol) = balancing_data.dim();
        InputError::check_sizes(b_nrow, container.population_size())?;
        let mut adjusted_data = Matrix::new(balancing_data.data(), b_nrow);

        for i in 0..b_nrow {
            let p = container.probabilities()[i];
            for j in 0..b_ncol {
                adjusted_data[(i, j)] /= p;
            }
        }

        Ok(CubeMethod {
            container,
            variant,
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data,
            candidate_data: Matrix::from_value(0.0, (b_ncol, b_ncol + 1)),
        })
    }
    #[inline]
    fn sample(&mut self) -> Result<Vec<usize>, SamplingError> {
        Ok(self.run().get_sorted_sample().to_vec())
    }
    #[inline]
    pub fn run(&mut self) -> &mut Self {
        self.run_flight().run_landing()
    }
    fn run_flight(&mut self) -> &mut Self {
        let b_cols = self.adjusted_data.ncol();
        assert_eq!(b_cols, self.candidate_data.nrow());

        while self.container.indices().len() > b_cols {
            self.variant
                .select_units(&mut self.candidates, &mut self.container, b_cols + 1);
            self.set_candidate_data().update_probabilities();
        }

        self
    }
    fn run_landing(&mut self) -> &mut Self {
        let b_cols = self.adjusted_data.ncol();
        assert!(
            self.container.indices().len() <= b_cols,
            "landing phase committed early: {} units remaining, with {} cols",
            self.container.indices().len(),
            b_cols,
        );

        while self.container.indices().len() > 1 {
            let number_of_remaining_units = self.container.indices().len();
            self.candidate_data
                .resize((number_of_remaining_units - 1, number_of_remaining_units));

            self.candidates.clear();
            self.candidates
                .extend_from_slice(self.container.indices().list());
            self.set_candidate_data().update_probabilities();
        }

        self.container
            .update_last_unit()
            .expect("last unit to be decided");

        self
    }
    #[inline]
    fn set_candidate_data(&mut self) -> &mut Self {
        let b_cols = self.candidates.len() - 1;
        assert_eq!(self.candidate_data.dim(), (b_cols, self.candidates.len()));

        for (i, &id) in self.candidates.iter().enumerate() {
            for j in 0..b_cols {
                self.candidate_data[(j, i)] = self.adjusted_data[(id, j)];
            }
        }

        self
    }
    #[inline]
    fn update_probabilities(&mut self) {
        let uvec = find_vector_in_null_space(&mut self.candidate_data);
        let mut lambdas = (f64::MAX, f64::MAX);

        for (prob, &uval) in self
            .candidates
            .iter()
            .map(|&id| self.container.probabilities()[id])
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

        let lambda = if self
            .container
            .rng()
            .one_of_f64(lambdas.0, lambdas.1)
            .unwrap()
        {
            lambdas.0
        } else {
            -lambdas.1
        };

        for (i, &id) in self.candidates.iter().enumerate() {
            self.container
                .add_probability_and_decide(id, lambda * uvec[i])
                .expect("id to update");
        }
    }
    #[inline]
    pub fn get_sample(&mut self) -> &[usize] {
        self.container.sample().get()
    }
    #[inline]
    pub fn get_sorted_sample(&mut self) -> &[usize] {
        self.container.sample_mut().sort().get()
    }
}

impl<'a, R> CubeMethodVariant<'a, R> for VariantCube
where
    R: RandomNumberGenerator + ?Sized,
{
    #[inline]
    fn new(
        rng: &'a mut R,
        options: &'a SampleOptions<'a>,
    ) -> Result<CubeMethod<'a, R, Self>, SamplingError> {
        CubeMethod::new(SampleContainer::new(rng, options)?, VariantCube {})
    }
    #[inline]
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &mut SampleContainer<'a, R>,
        n_units: usize,
    ) {
        assert!(container.indices().len() >= n_units);
        candidates.clear();
        candidates.extend_from_slice(&container.indices().list()[0..n_units]);
    }
}

impl<'a, R> CubeMethodVariant<'a, R> for VariantLocalCube
where
    R: RandomNumberGenerator + ?Sized,
{
    #[inline]
    fn new(
        rng: &'a mut R,
        options: &'a SampleOptions<'a>,
    ) -> Result<CubeMethod<'a, R, Self>, SamplingError> {
        let balancing = options.check_balancing()?.balancing().unwrap();
        let container = SampleContainer::new_with_tree(rng, options)?;
        let searcher = Searcher::new(
            container.tree().unwrap(),
            InputError::into_nonzero_usize(balancing.ncol())?,
        );

        CubeMethod::new(container, VariantLocalCube { searcher })
    }
    #[inline]
    fn select_units(
        &mut self,
        candidates: &mut Vec<usize>,
        container: &mut SampleContainer<'a, R>,
        n_units: usize,
    ) {
        assert!(n_units > 1);
        assert!(container.indices().len() >= n_units);
        candidates.clear();

        if container.indices().len() == n_units {
            candidates.extend_from_slice(container.indices().list());
            return;
        }

        // Draw the first unit at random
        let id1 = *container.indices_draw().unwrap();
        candidates.push(id1);

        // Find the neighbours of this first unit
        self.searcher
            .find_neighbours_of_id(container.tree().unwrap(), id1)
            .unwrap();

        // Add all neighbours, if no equals
        if self.searcher.neighbours().len() == n_units - 1 {
            candidates.extend_from_slice(self.searcher.neighbours());
            return;
        }

        let mut i: usize = 0;
        let maximum_distance = self
            .searcher
            .distance_k(self.searcher.neighbours().len() - 1);

        // Add all neighbours that are not on maximum distance
        while i < n_units - 1 && self.searcher.distance_k(i) < maximum_distance {
            let id = self.searcher.neighbours()[i];
            candidates.push(id);
            i += 1;
        }

        // Randomly add neighbours on the maximum distance
        for k in srs::sample(
            container.rng(),
            n_units - candidates.len(),
            self.searcher.neighbours().len() - i,
        )
        .unwrap()
        .iter()
        {
            candidates.push(self.searcher.neighbours()[i + k]);
        }
    }
}

pub struct CubeStratified<'a, R, T>
where
    R: RandomNumberGenerator + ?Sized,
    T: CubeMethodVariant<'a, R>,
{
    cube: CubeMethod<'a, R, T>,
    strata: HashMap<i64, Vec<usize>, FxSeededState>,
    strata_vec: &'a [i64],
}

pub trait CubeStratifiedVariant<'a, R>: CubeMethodVariant<'a, R>
where
    R: RandomNumberGenerator + ?Sized,
{
    fn reset_to(
        &mut self,
        container: &mut SampleContainer<R>,
        ids: &mut [usize],
        n_neighbours: usize,
    );
}

/// Draw a sample using the cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// For fixed sized samples, the first auxilliary variable should be the probability vector.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10);
/// let s = SampleOptions::new(&p)?.set_balancing(&bal_m)?.sample(&mut rng, cube)?;
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
#[inline]
pub fn cube<'a, R>(rng: &'a mut R, options: &SampleOptions<'a>) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    VariantCube::new(rng, options)?.sample()
}

/// Draw a sample using the stratified cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// The first auxilliary variable should not be the probability vector.
/// For fixed sized samples, the probabilities in each strata must be integer.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2; 10];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10);
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SampleOptions::new(&p)?.set_balancing(&bal_m)?;
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
#[inline]
pub fn cube_stratified<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
    strata: &'a [i64],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    let balancing_data = options.check_balancing()?.balancing().unwrap();
    let probabilities = options.probabilities();

    let seed = rng.rusize();
    let container = SampleContainer::new(rng, options)?;

    let mut cs = CubeStratified {
        cube: CubeMethod {
            container,
            variant: VariantCube {},
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data: Matrix::from_value(
                0.0,
                (balancing_data.nrow(), balancing_data.ncol() + 1),
            ),
            candidate_data: Matrix::from_value(
                0.0,
                (balancing_data.ncol() + 1, balancing_data.ncol() + 2),
            ),
        },
        strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
            probabilities.len() / 10,
            FxSeededState::with_seed(seed),
        ),
        strata_vec: strata,
    };

    cs.prepare().map(|s| s.sample())
}

/// Draw a sample using the local cube method.
/// The sample is balanced on the provided auxilliary variables in `balancing`.
/// the sample is spatially balanced on the provided auxilliary variables in `auxiliaries`.
/// For fixed sized samples, the first auxilliary variable should be the probability vector.
///
/// # Examples
/// ```
/// use envisim_samplr::cube_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10);
/// let spr_m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let s = SampleOptions::new(&p)?
///     .set_balancing(&bal_m)?
///     .set_spreading(&spr_m)?
///     .sample(&mut rng, local_cube)?;
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
///
/// Grafström, A., & Tillé, Y. (2013).
/// Doubly balanced spatial sampling with spreading and restitution of auxiliary totals.
/// Environmetrics, 24(2), 120-131.
/// <https://doi.org/10.1002/env.2194>
#[inline]
pub fn local_cube<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    VariantLocalCube::new(rng, options)?.sample()
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
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2; 10];
/// let bal_m = Matrix::from_vec(vec![
///     0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
///     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
/// ], 10);
/// let spr_m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10);
/// let strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
/// let options = SampleOptions::new(&p)?.set_balancing(&bal_m)?.set_spreading(&spr_m)?;
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
#[inline]
pub fn local_cube_stratified<'a, R>(
    rng: &'a mut R,
    options: &SampleOptions<'a>,
    strata: &'a [i64],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomNumberGenerator + ?Sized,
{
    let balancing_data = options.check_balancing()?.balancing().unwrap();
    let probabilities = options.probabilities();

    let seed = rng.rusize();
    let container = SampleContainer::new_with_tree(rng, options)?;
    let searcher = Searcher::new(
        container.tree().unwrap(),
        InputError::into_nonzero_usize(balancing_data.ncol() + 1)?,
    );

    let mut cs = CubeStratified {
        cube: CubeMethod {
            container,
            variant: VariantLocalCube { searcher },
            candidates: Vec::<usize>::with_capacity(20),
            adjusted_data: Matrix::from_value(
                0.0,
                (balancing_data.nrow(), balancing_data.ncol() + 1),
            ),
            candidate_data: Matrix::from_value(
                0.0,
                (balancing_data.ncol() + 1, balancing_data.ncol() + 2),
            ),
        },
        strata: HashMap::<i64, Vec<usize>, FxSeededState>::with_capacity_and_hasher(
            probabilities.len() / 10,
            FxSeededState::with_seed(seed),
        ),
        strata_vec: strata,
    };

    cs.prepare().map(|s| s.sample())
}

impl<'a, R> CubeStratifiedVariant<'a, R> for VariantCube
where
    R: RandomNumberGenerator + ?Sized,
{
    #[inline]
    fn reset_to(
        &mut self,
        container: &mut SampleContainer<R>,
        ids: &mut [usize],
        _n_neighbours: usize,
    ) {
        container.indices_mut().clear();
        for id in ids.iter() {
            container.indices_mut().insert(*id).unwrap();
        }
    }
}

impl<'a, R> CubeStratifiedVariant<'a, R> for VariantLocalCube
where
    R: RandomNumberGenerator + ?Sized,
{
    #[inline]
    fn reset_to(
        &mut self,
        container: &mut SampleContainer<R>,
        ids: &mut [usize],
        n_neighbours: usize,
    ) {
        self.searcher
            .set_n_neighbours(NonZeroUsize::new(n_neighbours).unwrap());

        container.indices_mut().clear();
        container
            .reset_tree(ids)
            .expect("tree should be resettable");

        for id in ids.iter() {
            container.indices_mut().insert(*id).unwrap();
        }
    }
}

impl<'a, R, T> CubeStratified<'a, R, T>
where
    R: RandomNumberGenerator + ?Sized,
    T: CubeStratifiedVariant<'a, R>,
{
    #[inline]
    fn prepare(&mut self) -> Result<&mut Self, SamplingError> {
        InputError::check_sizes(self.strata_vec.len(), self.cube.container.population_size())?;
        let balancing_data = self.cube.container.options().balancing().unwrap();
        let probabilities = self.cube.container.options().probabilities();

        for i in 0..self.cube.container.probabilities().len() {
            if !self.cube.container.indices().contains(i) {
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
            self.cube.adjusted_data[(i, balancing_data.ncol())] = 1.0;
            for j in 0..balancing_data.ncol() {
                self.cube.adjusted_data[(i, j)] = balancing_data[(i, j)] / probabilities[i];
            }
        }

        Ok(self)
    }
    #[inline]
    fn sample(&mut self) -> Vec<usize> {
        self.flight_per_stratum();
        if self.strata.is_empty() {
            return self.cube.get_sorted_sample().to_vec();
        }
        self.flight_on_full();
        if self.cube.container.indices().is_empty() {
            return self.cube.get_sorted_sample().to_vec();
        }
        self.landing_per_stratum();
        self.cube.get_sorted_sample().to_vec()
    }
    #[inline]
    fn flight_per_stratum(&mut self) {
        let mut removable_stratums = Vec::<i64>::new();
        for (stratum_key, stratum) in self.strata.iter_mut() {
            self.cube.variant.reset_to(
                &mut self.cube.container,
                stratum,
                self.cube.adjusted_data.ncol() + 1,
            );

            self.cube.run_flight();

            if self.cube.container.indices().is_empty() {
                removable_stratums.push(*stratum_key);
                continue;
            }

            stratum.clear();
            stratum.extend_from_slice(self.cube.container.indices().list());
        }

        for key in removable_stratums.iter() {
            self.strata.remove(key);
        }
    }
    #[inline]
    fn flight_on_full(&mut self) {
        let balancing_data = self.cube.container.options().balancing().unwrap();

        self.cube.adjusted_data.resize((
            balancing_data.nrow(),
            balancing_data.ncol() + self.strata.len(),
        ));
        self.cube.candidate_data.resize((
            self.cube.adjusted_data.ncol(),
            self.cube.adjusted_data.ncol() + 1,
        ));

        let mut all_units = Vec::<usize>::new();

        for (si, (_, stratum)) in self.strata.iter().enumerate() {
            all_units.extend_from_slice(stratum);

            for &id in stratum.iter() {
                self.cube.adjusted_data[(id, si + balancing_data.ncol())] = 1.0;
            }
        }

        self.cube.variant.reset_to(
            &mut self.cube.container,
            &mut all_units,
            self.cube.adjusted_data.ncol() + 1,
        );

        self.cube.run_flight();

        // Fix stratas
        self.strata.clear();

        for &id in self.cube.container.indices().list().iter() {
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
    #[inline]
    fn landing_per_stratum(&mut self) {
        let balancing_data = self.cube.container.options().balancing().unwrap();
        let probabilities = self.cube.container.options().probabilities();

        self.cube
            .adjusted_data
            .resize((balancing_data.nrow(), balancing_data.ncol() + 1));
        self.cube.candidate_data.resize((
            self.cube.adjusted_data.ncol(),
            self.cube.adjusted_data.ncol() + 1,
        ));

        for (_key, stratum) in self.strata.iter_mut() {
            for &id in stratum.iter() {
                self.cube.adjusted_data[(id, 0)] = 1.0;
                for j in 0..balancing_data.ncol() {
                    self.cube.adjusted_data[(id, j + 1)] =
                        balancing_data[(id, j)] / probabilities[id];
                }
            }

            self.cube.variant.reset_to(
                &mut self.cube.container,
                stratum,
                self.cube.adjusted_data.ncol() + 1,
            );

            self.cube.run_landing();
        }
    }
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
    use super::*;
    use envisim_test_utils::*;

    #[test]
    fn null() {
        let mut mat1 = Matrix::new(
            &[
                1.0, 2.0, 3.0, 1.0, 5.0, 10.0, 1.0, 5.0, 10.0, 1.0, 5.0, 10.0,
            ],
            3,
        );
        mat1.reduced_row_echelon_form();
        assert!(mat1.data() == [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        let mat1_nullvec = find_vector_in_null_space(&mut mat1);
        assert_fvec(&mat1.prod_vec(&mat1_nullvec), &[0.0, 0.0, 0.0]);

        let mut mat2 = Matrix::new(
            &[
                1.0, 2.0, 3.0, 1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 1.0, 5.0, 11.0,
            ],
            3,
        );
        mat2.reduced_row_echelon_form();
        assert!(&mat2.data()[0..9] == vec![1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_fvec(
            &mat2.data()[9..12], // col 3
            &[-2.5, 1.833333333333333, 0.166666666666667],
        );
        let mat2_nullvec = find_vector_in_null_space(&mut mat2);
        assert_fvec(&mat2.prod_vec(&mat2_nullvec), &[0.0, 0.0, 0.0]);
    }
}
