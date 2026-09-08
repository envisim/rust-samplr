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

//! Spatial balance measures

use std::iter::once;
use std::num::NonZeroUsize;

use envisim_utils::kd_tree::searcher::{
    NearestNeighbourSearcher,
    NeighbourView,
};
use envisim_utils::kd_tree::{
    Tree,
    TreeError,
};
use envisim_utils::matrix::{
    Matrix,
    MatrixDims,
    MatrixRef,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    ProbabilitiesSpec,
    SpreadingOptions,
};
use envisim_utils::utils::{
    ConstructableDataView,
    DataView,
    DataViewMut,
    Number,
    PointSet,
};
use num_traits::ToPrimitive;
use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};
use thiserror::Error;

/// Calculates the pi-sums for each voronoi cell
fn voronoi_pi_sum<PO, P, BAL, I>(
    opts: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    sample: I,
) -> Result<FxHashMap<PO::Id, P::Value>, SpatialBalanceError>
where
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id, Value = f64>,
    I: ExactSizeIterator<Item = PO::Id> + Clone,
{
    let data = opts.spreading().data();
    let sample_size = sample.len();
    let mut pi_sums =
        FxHashMap::<P::Id, P::Value>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for id in sample.clone() {
        let p = opts
            .probabilities()
            .get_real(id)
            .ok_or(SpatialBalanceError::InvalidId)?;
        pi_sums
            .insert(id, p)
            .map_or(Ok(()), |_| Err(SpatialBalanceError::DuplicateId))?;
    }

    let tree = Tree::from_iter(opts.spreading(), sample)?;
    let mut searcher = NearestNeighbourSearcher::new(data);

    for id in data.ids() {
        // Units in voronoi means have already been handled
        if pi_sums.contains_key(&id) {
            continue;
        }

        let p = opts
            .probabilities()
            .get_real(id)
            .ok_or(SpatialBalanceError::InvalidId)?;

        searcher.reset_from_point(data.coords(id).expect("id to exist in data by iter"));
        searcher.search(&tree).expect("search to find a unit");

        let share = p / searcher
            .neighbours()
            .len()
            .to_f64()
            .expect("limited by pop size, which is expected to convert to f64");

        for n in searcher.neighbours() {
            *pi_sums
                .get_mut(n.id())
                .expect("neighbours to exist amongst means") += share;
        }
    }

    Ok(pi_sums)
}

/// Calculate the voronoi means of `data`.
/// `sample` is an iterator over sample indices
#[expect(clippy::type_complexity, reason = "Ok complexity")]
fn voronoi_means<PO, P, BAL, I>(
    opts: &SamplingOptions<PO, SpreadingOptions<P>, BAL>,
    sample: I,
    balance_probabilities: bool,
) -> Result<FxHashMap<PO::Id, Box<[P::Value]>>, SpatialBalanceError>
where
    PO: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = PO::Id, Value = f64>,
    I: ExactSizeIterator<Item = PO::Id> + Clone,
{
    let data = opts.spreading().data();
    let sample_size = sample.len();
    let data_cols = data.dimensions().get();
    let mut means =
        FxHashMap::<P::Id, Box<[P::Value]>>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for id in sample.clone() {
        if !data.contains(id) {
            return Err(SpatialBalanceError::InvalidId);
        }
        let p = opts
            .probabilities()
            .get_real(id)
            .ok_or(SpatialBalanceError::InvalidId)?;
        let p_factor = (1.0 - p) / p;
        let id_mean: Box<[f64]> = if balance_probabilities {
            (0..data_cols)
                .map(|k| {
                    p_factor *
                     // SAFETY: id an k guaranteed to be in set
                     unsafe {data.coord_unchecked(id, k)}
                })
                .chain(once(p_factor))
                .collect()
        } else {
            (0..data_cols)
                .map(|k| {
                    p_factor *
                     // SAFETY: id an k guaranteed to be in set
                     unsafe {data.coord_unchecked(id, k)}
                })
                .collect()
        };
        means
            .insert(id, id_mean)
            .map_or(Ok(()), |_| Err(SpatialBalanceError::DuplicateId))?;
    }

    let tree = Tree::from_iter(opts.spreading(), sample)?;
    let mut searcher = NearestNeighbourSearcher::new(data);

    for id in data.ids() {
        // Units in voronoi means have already been handled
        if means.contains_key(&id) {
            continue;
        }

        searcher.reset_from_point(data.coords(id).expect("id to exist in data by iter"));
        searcher.search(&tree).expect("search to find a unit");

        let share = searcher
            .neighbours()
            .len()
            .to_f64()
            .expect("limited by pop size, which is expected to convert to f64");

        for &n in searcher.neighbours() {
            let mean = means
                .get_mut(n.id())
                .expect("neighbours to exist amongst means");

            for (j, m) in mean.iter_mut().enumerate().take(data_cols) {
                // SAFETY: id and j guaranteed to be in set
                *m -= unsafe { data.coord_unchecked(id, j) } / share;
            }

            if balance_probabilities {
                mean[data_cols] -= 1.0 / share;
            }
        }
    }

    Ok(means)
}

/// Calculate the norm matrix of `data`
fn norm_matrix<P>(data: P, balance_probabilities: bool) -> Matrix<P::Value>
where
    P: PointSet<Value = f64>,
{
    let data_cols = data.dimensions().get();
    let cols = data
        .dimensions()
        .saturating_add(usize::from(balance_probabilities));
    let mut norm_matrix = Matrix::from_value(0.0, MatrixDims::new(cols, cols));

    for id in data.ids() {
        // Last column as probs column
        if balance_probabilities {
            norm_matrix[(data.dimensions().get(), data.dimensions().get())] += 1.0;
        }

        for i in 0..data_cols {
            // SAFETY: id and i guaranteed to be in set
            let vi = unsafe { *data.coord_unchecked(id, i) };
            norm_matrix[(i, i)] += vi.powi(2);

            if balance_probabilities {
                norm_matrix[(data_cols, i)] += vi;
                norm_matrix[(i, data_cols)] += vi;
            }

            for j in 0..i {
                // SAFETY: id and i guaranteed to be in set
                let v = vi * unsafe { *data.coord_unchecked(id, j) };
                norm_matrix[(i, j)] += v;
                norm_matrix[(j, i)] += v;
            }
        }
    }

    norm_matrix
}

/// Calulates the energy distance between a population and a sample.
/// # References
/// Grafström & Prentius (2026). Distributionally balanced sampling designs. Biometrics, 82(3).
pub struct EnergyDistance<PH, DT> {
    /// Mean unit-distances, i.e. the average distance of each unit to the population.
    phis: PH,
    /// The population spread, multiplied by `sample_size`.
    u_spread: f64,
    /// The sample size.
    sample_size: NonZeroUsize,
    /// The data
    data: DT,
}

impl<PH, DT> EnergyDistance<PH, DT>
where
    PH: DataView<Value = f64>,
    DT: PointSet<Id = PH::Id, Value = f64>,
{
    /// Constructs a new Energy distance object
    /// # Errors
    /// Returns an error if units in `probabilities` does not exist in `data`.
    /// # Panics
    /// Panics if `sample_size` cannot be converted into `f64`.
    #[inline]
    pub fn new<PO>(
        probabilities: PO,
        data: DT,
        sample_size: NonZeroUsize,
    ) -> Result<Self, SpatialBalanceError>
    where
        PH: DataViewMut,
        PO: ProbabilitiesSpec<Id = PH::Id, Value: Number, Real = f64>
            + ConstructableDataView<ConstructableContainer<f64> = PH>,
    {
        let nn = sample_size
            .get()
            .to_f64()
            .expect("sample size converts to f64");
        let ids: Box<[PH::Id]> = probabilities.ids().collect();
        let mut phis: PH = probabilities.iter_map(|(id, _)| (id, 0.0));
        let mut u_spread = 0.0;

        for (k, &id1) in ids.entries() {
            // Panic here should be impossible
            let p_1 = probabilities.get_real(id1).expect("id1 to exist");
            let mut phi_1 = *phis.get(id1).expect("id1 to exist");
            for &id2 in ids.iter().skip(k + 1) {
                let p_2 = probabilities.get_real(id2).expect("id2 to exist");
                let dist = data
                    .sq_distance_between(id1, id2)
                    .ok_or(SpatialBalanceError::InvalidId)?
                    .sqrt();
                phi_1 += dist * p_2;
                *phis.get_mut(id2).expect("id2 to exist") += dist * p_1;
            }
            *phis.get_mut(id1).expect("id1 to exist") = phi_1 / nn;
            u_spread += phi_1 * p_1;
        }

        Ok(Self {
            phis,
            u_spread,
            sample_size,
            data,
        })
    }
    /// Returns the sample size as `f64` as a convenience method.
    /// # Panics
    /// Panics if `sample_size` cannot be converted into `f64`.
    #[inline]
    fn nn(&self) -> f64 {
        self.sample_size
            .get()
            .to_f64()
            .expect("sample size converts to f64")
    }
    /// Returns the average distancee of unit `id` to the rest of the population.
    #[inline]
    pub fn phi(&self, id: PH::Id) -> Option<f64> { self.phis.get(id).copied() }
    /// Returns the population spread.
    /// See [`EnergyDistance::u_spread_n`].
    #[inline]
    pub fn u_spread(&self) -> f64 { self.u_spread / self.nn() }
    /// Returns the population spread, or the average pairwise distances between units in the
    /// population, multiplied by the sample size.
    #[inline]
    pub fn u_spread_n(&self) -> f64 { self.u_spread }
    /// Returns the energy distance between a sample and the population.
    /// See [`EnergyDistance::energy_distance_n`].
    #[expect(clippy::missing_errors_doc, reason = "referred to other function")]
    #[inline]
    pub fn energy_distance<I>(&self, sample: I) -> Result<f64, SpatialBalanceError>
    where
        I: Iterator<Item = PH::Id> + Clone,
    {
        self.energy_distance_n(sample).map(|ed| ed / self.nn())
    }
    /// Returns the energy distance between a sample and the population, multiplied by the sample
    /// size.
    /// # Errors
    /// Returns an error if any ID in the sample does not exist in the population.
    #[inline]
    pub fn energy_distance_n<I>(&self, sample: I) -> Result<f64, SpatialBalanceError>
    where
        I: Iterator<Item = PH::Id> + Clone,
    {
        let mut s_spread = 0.0;
        let mut i_spread = 0.0;

        let mut outer = sample;

        while let Some(id1) = outer.next() {
            let phi1 = self.phi(id1).ok_or(SpatialBalanceError::InvalidId)?;
            i_spread += phi1;

            let inner = outer.clone();
            for id2 in inner {
                s_spread += self
                    .data
                    .sq_distance_between(id1, id2)
                    .ok_or(SpatialBalanceError::InvalidId)?
                    .sqrt();
            }
        }

        s_spread *= 2.0 / self.nn();
        i_spread *= 2.0;
        Ok(i_spread - s_spread - self.u_spread_n())
    }
    /// Returns the energy difference.
    /// See [`EnergyDistance::delta_n`].
    #[expect(clippy::missing_errors_doc, reason = "referred to other function")]
    #[inline]
    pub fn delta<I>(&self, sample: I, add: PH::Id, rem: PH::Id) -> Result<f64, SpatialBalanceError>
    where
        I: Iterator<Item = PH::Id>,
    {
        self.delta_n(sample, add, rem).map(|d| d / self.nn())
    }
    /// Returns the energy difference when adding `add` and removing `rem` from a sample.
    /// # Errors
    /// Returns an error if any id in the `sample`, `add` or `rem` does not exist in the population.
    #[inline]
    pub fn delta_n<I>(
        &self,
        sample: I,
        add: PH::Id,
        rem: PH::Id,
    ) -> Result<f64, SpatialBalanceError>
    where
        I: Iterator<Item = PH::Id>,
    {
        let i_delta = self
            .phi(add)
            .zip(self.phi(rem))
            .map(|(add, rem)| (add - rem) * 2.0)
            .ok_or(SpatialBalanceError::InvalidId)?;

        let mut s_delta = 0.0;
        for id in sample {
            if id == add {
                // Sample already contains add
                return Err(SpatialBalanceError::InvalidAddId);
            } else if id == rem {
                continue;
            }
            s_delta += self
                .data
                .sq_distance_between(id, rem)
                .ok_or(SpatialBalanceError::InvalidId)?
                .sqrt()
                - self
                    .data
                    .sq_distance_between(id, add)
                    .ok_or(SpatialBalanceError::InvalidId)?
                    .sqrt();
        }

        s_delta *= 2.0 / self.nn();
        Ok(i_delta + s_delta)
    }
}

/// Spatial balance error types
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum SpatialBalanceError {
    /// Error derived from [`Tree`]
    #[error("TreeError: {0}")]
    Tree(#[from] TreeError),
    /// Invalid ID, ID missing from collection
    #[error("Invalid ID (missing from collection)")]
    InvalidId,
    /// Invalid ID, ID is already in sample
    #[error("Invalid ID (cannot add an ID already in a sample)")]
    InvalidAddId,
    /// Invalid ID, duplicate ID
    #[error("Invalid ID (duplicate ID found)")]
    DuplicateId,
    /// Invalid sample size, sample size must be positive
    #[error("Invalid sample size (must be positive)")]
    InvalidSampleSize,
    /// Auxiliaries not invertible
    #[error("Auxiliaries not invertible")]
    SingularMatrix,
}

/// Provides methods for calculating the spatial balance of a sample
pub trait SpatialBalance<P>
where
    P: PointSet,
{
    /// Voronoi measure of spatial balance.
    ///
    /// # Examples
    /// ```
    /// # use envisim_estimate::spatial_balance::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let options = SamplingOptions::new(p)?.set_spreading(m)?;
    /// let s = [0, 3, 5, 8, 9];
    /// let sb = options.voronoi(&s)?;
    /// # Ok::<(), EstimationError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., & Schelin, L. (2014).
    /// How to select representative samples.
    /// Scandinavian Journal of Statistics, 41(2), 277-290.
    /// <https://doi.org/10.1111/sjos.12016>
    ///
    /// # Errors
    /// If `sample` contains duplicate ids
    fn voronoi<I>(&self, sample: I) -> Result<P::Value, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone;
    /// Local measure of spatial balance.
    ///
    /// # Examples
    /// ```
    /// # use envisim_estimate::spatial_balance::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let options = SamplingOptions::new(p)?.set_spreading(m)?;
    /// let s = [0, 3, 5, 8, 9];
    /// let sb = options.local(&s, true)?;
    /// # Ok::<(), EstimationError>(())
    /// ```
    ///
    /// # References
    /// Prentius, W., & Grafström, A. (2024).
    /// How to find the best sampling design: A new measure of spatial balance.
    /// Environmetrics, e2878.
    /// <https://doi.org/10.1002/env.2878>
    ///
    /// # Errors
    /// If `sample` contains duplicate ids
    fn local<I>(
        &self,
        sample: I,
        balance_probabilities: bool,
    ) -> Result<P::Value, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone;
    /// Energy distance between sample distribution and population.
    ///
    /// # Examples
    /// ```
    /// # use envisim_estimate::spatial_balance::*;
    /// # use envisim_utils::matrix::Matrix;
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let options = SamplingOptions::new(p)?.set_spreading(m)?;
    /// let s = [0, 3, 5, 8, 9];
    /// let sb = options.energy_distance(&s);
    /// # Ok::<(), EstimationError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., &  Prentius, W. (2026).
    /// Distributionally balanced sampling designs.
    /// Biometrics, 82(3).
    /// <https://doi.org/10.1093/biomtc/ujag124>
    ///
    /// # Errors
    /// Returns an error if any sample id does not exist in the population.
    fn energy_distance<I>(&self, sample: I) -> Result<f64, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone;
}

impl<PO, P, BAL> SpatialBalance<P> for SamplingOptions<PO, SpreadingOptions<P>, BAL>
where
    PO: ProbabilitiesSpec<Real = f64> + ConstructableDataView,
    P: PointSet<Id = PO::Id, Value = f64>,
{
    /// # Panics
    /// Panics if sample size cannot be converted to `f64`.
    #[inline]
    fn voronoi<I>(&self, sample: I) -> Result<P::Value, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone,
    {
        if sample.len() == 0 {
            return Ok(f64::NAN);
        }

        let sample_size = sample.len();
        let voronoi_pi = voronoi_pi_sum(self, sample)?;
        let result = voronoi_pi.values().map(|v| (v - 1.0).powi(2)).sum::<f64>()
            / sample_size.to_f64().expect("sample len to convert to f64");

        Ok(result)
    }
    /// # Panics
    /// Panics if sample size cannot be converted to `f64`.
    #[inline]
    fn local<I>(
        &self,
        sample: I,
        balance_probabilities: bool,
    ) -> Result<P::Value, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone,
    {
        if sample.len() == 0 {
            return Ok(f64::NAN);
        }

        let data = self.spreading().data();
        let cols = data
            .dimensions()
            .saturating_add(usize::from(balance_probabilities));
        let vor_means = voronoi_means(self, sample, balance_probabilities)?;

        // The gram matrix
        let inv_norm_matrix = norm_matrix(data, balance_probabilities)
            .inverse(self.eps())
            .ok_or(SpatialBalanceError::SingularMatrix)?;

        let result = vor_means
            .values()
            .map(|mean| {
                MatrixRef::new(mean, 1)
                    .expect("1 > 0")
                    .mul_mat(&inv_norm_matrix)
                    .expect("dimensions to match")
                    .mul_mat(&MatrixRef::new(mean, cols).expect("cols = vec.len"))
                    .expect("dimensions to match")[(0, 0)]
            })
            .sum::<f64>()
            / self
                .population_size()
                .get()
                .to_f64()
                .expect("population size to convert to f64");

        Ok(result.sqrt())
    }
    #[inline]
    fn energy_distance<I>(&self, sample: I) -> Result<f64, SpatialBalanceError>
    where
        I: ExactSizeIterator<Item = P::Id> + Clone,
    {
        let sample_size =
            NonZeroUsize::new(sample.len()).ok_or(SpatialBalanceError::InvalidSampleSize)?;
        let matrix = self.spreading().data();
        let ed = EnergyDistance::new(self.probabilities(), matrix, sample_size)?;
        ed.energy_distance(sample)
    }
}

#[cfg(test)]
mod test {
    use envisim_utils::test_utils::*;

    use super::*;

    #[test]
    fn ed_phi() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = MatrixRef::new(&m_data, nz(3)).unwrap();
        let (phi, _) = energy_distance_phi_equal(&data);
        let phi_res = vec![phi[&0], phi[&1], phi[&2]];
        let facit: Vec<f64> = vec![
            (2.0f64.sqrt() + 8.0f64.sqrt()) / 3.0f64,
            (2.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
            (8.0f64.sqrt() + 2.0f64.sqrt()) / 3.0f64,
        ];

        assert_vec!(phi_res, facit);
    }
    #[test]
    fn ed_internal() {
        let m_data: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let data = MatrixRef::new(&m_data, nz(3)).unwrap();
        let (phi, _) = energy_distance_phi_equal(&data);
        let dist = energy_distance_internal(&[1, 2], &data, &phi);
        let res: f64 = 2.0 * (phi[&1] + phi[&2]) / 2.0 - (2.0f64.sqrt() + 2.0f64.sqrt()) / 4.0;

        assert_delta!(dist, res);
    }

    #[test]
    fn test_voronoi() {
        let options = Data10::options_e();
        let sb = options.voronoi(&[0]).unwrap();
        assert_delta!(sb, (0.2f64 * 10.0 - 1.0).powi(2));
    }

    #[test]
    fn test_local() {
        let options = Data10::options_e();
        let sb = options.local(&[0], false).unwrap();
        assert_delta!(sb, 0.7515302, Epsilon::new(1e-7).unwrap());

        let sb = options.local(&[0, 1], false).unwrap();
        assert_delta!(sb, 0.6454327, Epsilon::new(1e-7).unwrap());
    }
}
