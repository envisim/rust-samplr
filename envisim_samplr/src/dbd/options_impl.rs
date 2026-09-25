// Copyright (C) 2026 Wilmer Prentius
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

//! Distributionally balanced design trait.
use std::num::NonZeroUsize;

use envisim_estimate::spatial_balance::EnergyDistance;
use envisim_utils::random::Rng;
use envisim_utils::sampling_options::{
    EqualProbabilities,
    ProbabilitiesSpec,
    SamplingOptions,
    SamplingOptionsRng,
    SpreadingOptions,
};
use envisim_utils::utils::{
    DataView,
    PointSet,
};
use num_traits::ToPrimitive;

use super::annealing::AnnealingDistributionalDesign;
use super::dbd_circular::{
    CircularConfiguration,
    DbdCircular,
};
use super::dbd_options::DistributionalDesignOptions;
use super::dbd_tc::{
    DbdTacticalConfiguration,
    TacticalConfiguration,
};
use super::tc_parameters::DbdConfiguration;
use crate::{
    SamplingError,
    SamplingResult,
};

/// Provides distributionally balanced sampling designs.
pub trait DistributionalDesigns<ID, R>
where
    R: Rng,
{
    /// Construct a distributionally balanced design in a circular configuration.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let opts = SamplingOptions::new_equal(10, 2)?.set_spreading(m)?;
    /// let dbd_opts = DistributionalDesignOptions::default();
    /// let dbd = opts.dbd_circular(&mut rng, dbd_opts)?;
    /// let s: Vec<usize> = dbd.draw(&mut rng).collect();
    /// assert_eq!(s.len(), 2);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if `sample_size` is 0.
    fn dbd_circular(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<CircularConfiguration<ID>>;
    /// Construct a distributionally balanced design using a tactical configuration.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// # use envisim_utils::matrix::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let m = Matrix::new(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
    /// let opts = SamplingOptions::new_equal(10, 2)?.set_spreading(m)?;
    /// let dbd_opts = DistributionalDesignOptions::default();
    /// let dbd = opts.dbd_tc(&mut rng, dbd_opts)?;
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if `sample_size` is 0.
    fn dbd_tc(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<TacticalConfiguration<ID>>;
}

impl<R, IDS, P> DistributionalDesigns<<EqualProbabilities<IDS> as DataView>::Id, R>
    for SamplingOptions<EqualProbabilities<IDS>, SpreadingOptions<P>>
where
    R: SamplingOptionsRng<EqualProbabilities>,
    EqualProbabilities<IDS>: ProbabilitiesSpec<Real = f64>,
    P: PointSet<Id = <EqualProbabilities<IDS> as DataView>::Id, Value = f64>,
{
    #[inline]
    fn dbd_circular(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<CircularConfiguration<P::Id>> {
        let spreading_data = self.spreading().data();
        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let ed = EnergyDistance::new(self.probabilities(), spreading_data, sample_size)?;

        let max_iter = self.max_iterations();
        let eps = self.eps();

        let mut v = match DbdCircular::new(ed, &dbs_options, eps) {
            Ok(v) => v,
            Err(c) => return Ok(c),
        };
        v.run(rng, max_iter);

        Ok(v.into_optimal_configuration())
    }
    #[inline]
    fn dbd_tc(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<TacticalConfiguration<P::Id>> {
        let spreading_data = self.spreading().data();
        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let ed = EnergyDistance::new(self.probabilities(), spreading_data, sample_size)?;
        let max_iter = self.max_iterations();
        let eps = self.eps();

        let mut v = match DbdTacticalConfiguration::new(rng, ed, &dbs_options, eps) {
            Ok(v) => v,
            Err(c) => return Ok(c),
        };
        v.run(rng, max_iter);

        Ok(v.into_optimal_configuration())
    }
}

/// Provides evalutors for distributionally balanced sampling designs
pub trait DistributionalDesignEvaluators<R>
where
    R: Rng,
{
    /// Runs the circular dbd until `to`, reporting the energy in `by` intervals.
    ///
    /// # Errors
    /// Returns an error if `sample_size` is 0.
    fn dbd_circular_evaluator(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: NonZeroUsize,
        by: NonZeroUsize,
    ) -> SamplingResult<Vec<f64>>;
    /// Runs the tactical configuration dbd until `to`, reporting the energy in `by` intervals.
    ///
    /// # Errors
    /// Returns an error if `sample_size` is 0.
    fn dbd_tc_evaluator(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: NonZeroUsize,
        by: NonZeroUsize,
    ) -> SamplingResult<Vec<f64>>;
}
impl<R, P> DistributionalDesignEvaluators<R>
    for SamplingOptions<EqualProbabilities, SpreadingOptions<P>>
where
    R: SamplingOptionsRng<EqualProbabilities>,
    P: PointSet<Id = usize, Value = f64>,
{
    #[inline]
    fn dbd_circular_evaluator(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: NonZeroUsize,
        by: NonZeroUsize,
    ) -> SamplingResult<Vec<f64>> {
        if to < by {
            return Err(SamplingError::MaxIterations(to));
        }

        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let spreading_data = self.spreading().data();
        let ed = EnergyDistance::new(self.probabilities(), spreading_data, sample_size)?;

        let sample_size_float = sample_size
            .get()
            .to_f64()
            .expect("sample_size to convert to f64");
        let eps = self.eps();
        #[expect(clippy::integer_division, reason = "no loss of precision")]
        let res_size = to.get() / by * 2;

        let Ok(mut v) = DbdCircular::new(ed, &dbs_options, eps) else {
            return Ok(vec![0.0; res_size]);
        };

        let mut res: Vec<f64> = Vec::with_capacity(res_size);

        let mut iters = by.get();
        while iters <= to.get() {
            v.run(rng, by);

            let optimal_conf = v.optimal_configuration();
            let mean = optimal_conf.average_energy();
            let mut sd = 0.0;

            for sid in 0..optimal_conf.tcp().n_samples().get() {
                let energy = optimal_conf.energy_of_sample_n(v.ed(), sid) / sample_size_float;
                sd += (energy - mean).powi(2);
            }

            sd = (sd
                / optimal_conf
                    .tcp()
                    .n_samples()
                    .get()
                    .to_f64()
                    .expect("n_samples to convert to f64"))
            .sqrt();
            res.push(mean);
            res.push(sd);
            iters += by.get();
        }

        Ok(res)
    }
    #[inline]
    fn dbd_tc_evaluator(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: NonZeroUsize,
        by: NonZeroUsize,
    ) -> SamplingResult<Vec<f64>> {
        if to < by {
            return Err(SamplingError::MaxIterations(to));
        }

        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let spreading_data = self.spreading().data();
        let ed = EnergyDistance::new(self.probabilities(), spreading_data, sample_size)?;

        let sample_size_float = sample_size
            .get()
            .to_f64()
            .expect("sample_size to convert to f64");
        let eps = self.eps();
        #[expect(clippy::integer_division, reason = "no loss of precision")]
        let res_size = to.get() / by * 2;

        let Ok(mut v) = DbdTacticalConfiguration::new(rng, ed, &dbs_options, eps) else {
            return Ok(vec![0.0; res_size]);
        };

        let mut res: Vec<f64> = Vec::with_capacity(res_size);
        let n_buckets_float = v
            .tcp()
            .n_samples()
            .get()
            .to_f64()
            .expect("n_samples to convert to f64");

        let mut iters = by.get();
        while iters <= to.get() {
            v.run(rng, by);

            let optimal_conf = v.optimal_configuration();
            let mean = optimal_conf.average_energy();
            let mut sd = 0.0;

            for sid in 0..optimal_conf.tcp().n_samples().get() {
                let energy = optimal_conf.energy_of_sample_n(v.ed(), sid) / sample_size_float;
                sd += (energy - mean).powi(2);
            }

            sd = (sd / n_buckets_float).sqrt();
            res.push(mean);
            res.push(sd);
            iters += by.get();
        }

        Ok(res)
    }
}
