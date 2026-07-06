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

use envisim_utils::kd_tree::Tree;
use envisim_utils::kd_tree::searcher::NearestNeighbourSearcher;
use envisim_utils::matrix::{
    Matrix,
    MatrixDims,
    MatrixRef,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    EqualProbabilityOptions,
    ProbabilityOptions,
    SamplingOptionsError,
    SpreadingOptions,
    UnequalProbabilityOptions,
};
use envisim_utils::utils::{
    Number,
    PointSet,
};
use num_traits::{
    ConstZero,
    ToPrimitive,
};
use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

pub use crate::error::EstimationError;
use crate::error::EstimationResult;

/// Calculates the pi-sums for each voronoi cell
fn voronoi_pi_sum<P, F>(
    opts: &SpreadingOptions<P>,
    sample: &[P::Id],
    prob: F,
) -> EstimationResult<FxHashMap<P::Id, P::Value>>
where
    P: PointSet<Value = f64>,
    F: Fn(P::Id) -> P::Value,
{
    let data = opts.data();
    let sample_size = sample.len();
    let mut pi_sums =
        FxHashMap::<P::Id, P::Value>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for &id in sample {
        let p = prob(id);
        pi_sums
            .insert(id, p)
            .map_or(Ok(()), |_| Err(EstimationError::InvalidSample))?;
    }

    let tree = Tree::new(opts, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(data);

    for id in data.ids() {
        // Units in voronoi means have already been handled
        if pi_sums.contains_key(&id) {
            continue;
        }

        searcher.reset_from_point(data.get_coords(id).expect("id to exist in data by iter"));
        searcher.search(&tree).expect("search to find a unit");

        let share = prob(id)
            / searcher
                .neighbours()
                .len()
                .to_f64()
                .expect("limited by pop size, which is expected to convert to f64");

        for n in searcher.neighbours() {
            *pi_sums
                .get_mut(&n.id())
                .expect("neighbours to exist amongst means") += share;
        }
    }

    Ok(pi_sums)
}

/// Calculate the voronoi means of `data`.
/// `sample` is an iterator over sample indices and their probability factor (1-pi)/pi
#[expect(clippy::type_complexity, reason = "Ok complexity")]
fn voronoi_means<P, F>(
    opts: &SpreadingOptions<P>,
    sample: &[P::Id],
    prob: F,
    balance_probabilities: bool,
) -> EstimationResult<FxHashMap<P::Id, Box<[P::Value]>>>
where
    P: PointSet<Value = f64>,
    F: Fn(P::Id) -> P::Value,
{
    let data = opts.data();
    let sample_size = sample.len();
    let data_cols = data.dimensions().get();
    let mut means =
        FxHashMap::<P::Id, Box<[P::Value]>>::with_capacity_and_hasher(sample_size, FxBuildHasher);

    for &id in sample {
        let p_factor = prob(id);
        let id_mean: Box<[f64]> = if balance_probabilities {
            (0..data_cols)
                .map(|k| p_factor * data.coord(id, k))
                .chain(once(p_factor))
                .collect()
        } else {
            (0..data_cols)
                .map(|k| p_factor * data.coord(id, k))
                .collect()
        };
        means
            .insert(id, id_mean)
            .map_or(Ok(()), |_| Err(EstimationError::InvalidSample))?;
    }

    let tree = Tree::new(opts, &mut sample.to_vec());
    let mut searcher = NearestNeighbourSearcher::new(data);

    for id in data.ids() {
        // Units in voronoi means have already been handled
        if means.contains_key(&id) {
            continue;
        }

        searcher.reset_from_point(data.get_coords(id).expect("id to exist in data by iter"));
        searcher.search(&tree).expect("search to find a unit");

        let share = searcher
            .neighbours()
            .len()
            .to_f64()
            .expect("limited by pop size, which is expected to convert to f64");

        for &n in searcher.neighbours() {
            let mean = means
                .get_mut(&n.id())
                .expect("neighbours to exist amongst means");

            for (j, m) in mean.iter_mut().enumerate().take(data_cols) {
                *m -= data.coord(id, j) / share;
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
            let vi = data.coord(id, i);
            norm_matrix[(i, i)] += vi.powi(2);

            if balance_probabilities {
                norm_matrix[(data_cols, i)] += vi;
                norm_matrix[(i, data_cols)] += vi;
            }

            for j in 0..i {
                let v = vi * data.coord(id, j);
                norm_matrix[(i, j)] += v;
                norm_matrix[(j, i)] += v;
            }
        }
    }

    norm_matrix
}

/// Returns (phi-vec, phi-sumish)
#[must_use]
#[inline]
fn energy_distance_phi_equal<P>(matrix: &P) -> (FxHashMap<P::Id, P::Value>, P::Value)
where
    P: PointSet<Value = f64>,
{
    let size = matrix.len().get();
    let mut phi = FxHashMap::<P::Id, P::Value>::with_capacity_and_hasher(size, FxBuildHasher);

    for (i, id1) in matrix.ids().enumerate() {
        let mut phi1 = <P::Value as ConstZero>::ZERO;
        for id2 in matrix.ids().take(i) {
            let dist = matrix.sq_distance_between(id1, id2).sqrt();
            phi1 += dist;
            *phi.get_mut(&id2).expect("id2 to exist in map") += dist;
        }
        phi.insert(id1, phi1);
    }

    let u_size = size.to_f64().expect("pop size to convert to f64");
    let mut u_spread = <P::Value as ConstZero>::ZERO;
    for val in phi.values_mut() {
        *val /= u_size;
        u_spread += *val;
    }

    (phi, u_spread / u_size)
}
/// Returns (phi-vec, phi-sumish)
#[must_use]
#[inline]
fn energy_distance_phi_unequal<P, I>(
    matrix: P,
    probabilities: I,
    s_size: f64,
) -> (FxHashMap<P::Id, P::Value>, P::Value)
where
    P: PointSet<Id = usize, Value = f64>,
    I: ExactSizeIterator<Item = f64>,
{
    let population_size = probabilities.len();
    assert!(
        population_size <= matrix.len().get(),
        "not able to map between prob indices and PointSet"
    );
    let mut phi =
        FxHashMap::<P::Id, P::Value>::with_capacity_and_hasher(population_size, FxBuildHasher);

    let probs: Vec<f64> = probabilities.map(|p| p / s_size).collect();

    for id1 in 0..population_size {
        let mut phi1 = <P::Value as ConstZero>::ZERO;

        for id2 in 0..id1 {
            let dist = matrix.sq_distance_between(id1, id2).sqrt();
            phi1 += dist * probs[id2];
            *phi.get_mut(&id2).expect("id2 to exist in map") += dist * probs[id1];
        }

        phi.insert(id1, phi1);
    }

    let mut u_spread = <P::Value as ConstZero>::ZERO;
    for (&id, val) in &mut phi {
        // Second div. by s_size goes here
        u_spread += *val * probs[id];
    }

    (phi, u_spread)
}

/// Returns 2 E||X-Z|| - E||X-X'||
#[must_use]
#[inline]
fn energy_distance_internal<P>(
    sample: &[P::Id],
    matrix: P,
    phi: &FxHashMap<P::Id, P::Value>,
) -> P::Value
where
    P: PointSet<Value = f64>,
{
    let s_size = sample
        .len()
        .to_f64()
        .expect("sample size to convert to f64");
    let mut s_spread = <P::Value as ConstZero>::ZERO;
    let mut inter_spread = <P::Value as ConstZero>::ZERO;

    for (i, id1) in sample.iter().enumerate() {
        inter_spread += phi[id1];

        // Iterate over 0..i
        for id2 in sample.iter().take(i) {
            let dist = matrix.sq_distance_between(*id1, *id2).sqrt();
            s_spread += 2.0 * dist;
        }
    }

    inter_spread /= s_size;
    s_spread /= s_size.powi(2);
    inter_spread * 2.0 - s_spread
}

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
    fn voronoi(&self, sample: &[P::Id]) -> EstimationResult<P::Value>;
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
    fn local(&self, sample: &[P::Id], balance_probabilities: bool) -> EstimationResult<P::Value>;
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
    #[must_use]
    fn energy_distance(&self, sample: &[P::Id]) -> P::Value;
}

impl<P, BAL> SpatialBalance<P>
    for SamplingOptions<EqualProbabilityOptions, SpreadingOptions<P>, BAL>
where
    P: PointSet<Value = f64>,
{
    #[inline]
    fn voronoi(&self, sample: &[P::Id]) -> EstimationResult<P::Value> {
        if sample.is_empty() {
            return Ok(f64::NAN);
        }

        let p = self.probabilities().as_real();
        let voronoi_pi = voronoi_pi_sum(self.spreading(), sample, |_| p)?;
        let result = voronoi_pi.values().map(|v| (v - 1.0).powi(2)).sum::<f64>()
            / sample.len().to_f64().expect("sample len to convert to f64");

        Ok(result)
    }
    #[inline]
    fn local(&self, sample: &[P::Id], balance_probabilities: bool) -> EstimationResult<P::Value> {
        if sample.is_empty() {
            return Ok(f64::NAN);
        }

        let data = self.spreading().data();
        let cols = data
            .dimensions()
            .saturating_add(usize::from(balance_probabilities));
        let p = self.probabilities().as_real();
        let p_factor = (1.0 - p) / p;
        let voronoi_means = voronoi_means(
            self.spreading(),
            sample,
            |_| p_factor,
            balance_probabilities,
        )?;

        // The gram matrix
        let inv_norm_matrix = norm_matrix(data, balance_probabilities)
            .inverse(self.eps())
            .ok_or(SamplingOptionsError::InvalidSpreading)?;

        let result = voronoi_means
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
    fn energy_distance(&self, sample: &[P::Id]) -> P::Value {
        let matrix = self.spreading().data();
        let (phi, u_spread) = energy_distance_phi_equal(matrix);
        let edi = energy_distance_internal(sample, matrix, &phi);
        edi - u_spread
    }
}

impl<UPO, P, BAL> SpatialBalance<P>
    for SamplingOptions<UnequalProbabilityOptions<UPO>, SpreadingOptions<P>, BAL>
where
    UPO: ProbabilityOptions<Real = f64>,
    P: PointSet<Id = usize, Value = f64>,
{
    #[inline]
    fn voronoi(&self, sample: &[P::Id]) -> EstimationResult<P::Value> {
        if sample.is_empty() {
            return Ok(f64::NAN);
        }

        let probs: Box<[f64]> = self.probabilities().iter_real().collect();
        let voronoi_pi = voronoi_pi_sum(self.spreading(), sample, |id| probs[id])?;
        let result = voronoi_pi.values().map(|v| (v - 1.0).powi(2)).sum::<f64>()
            / sample.len().to_f64().expect("sample len to convert to f64");

        Ok(result)
    }
    #[inline]
    fn local(&self, sample: &[P::Id], balance_probabilities: bool) -> EstimationResult<P::Value> {
        if sample.is_empty() {
            return Ok(f64::NAN);
        }

        let data = self.spreading().data();
        let cols = data
            .dimensions()
            .saturating_add(usize::from(balance_probabilities));
        let probs: Box<[f64]> = self.probabilities().iter_real().collect();
        let voronoi_means = voronoi_means(
            self.spreading(),
            sample,
            |id| {
                let p = probs[id];
                (1.0 - p) / p
            },
            balance_probabilities,
        )?;

        // The gram matrix
        let inv_norm_matrix = norm_matrix(data, balance_probabilities)
            .inverse(self.eps())
            .ok_or(SamplingOptionsError::InvalidSpreading)?;

        let result = voronoi_means
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
    /// Requires that [`PointSet`] is able to map probability indices
    #[inline]
    fn energy_distance(&self, sample: &[P::Id]) -> P::Value {
        let matrix = self.spreading().data();
        let s_size = sample
            .len()
            .to_f64()
            .expect("sample size to convert to f64");
        let (phi, u_spread) =
            energy_distance_phi_unequal(matrix, self.probabilities().iter_real(), s_size);
        let edi = energy_distance_internal(sample, matrix, &phi);
        edi - u_spread
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
