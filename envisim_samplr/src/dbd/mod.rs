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

//! Distributionally balanced designs

mod annealing;
mod dbd_circular;
mod dbd_options;
mod dbd_tc;
mod energy_distance;
mod tc_parameters;
mod utils;

use std::num::NonZeroUsize;

use annealing::AnnealingDistributionalDesign;
pub use dbd_circular::CircularConfiguration;
use dbd_circular::DbdCircular;
pub use dbd_options::DistributionalDesignOptions;
use dbd_tc::DbdTacticalConfiguration;
pub use dbd_tc::TacticalConfiguration;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    ProbabilitySpecEqual,
    SamplingOptions,
};
use envisim_utils::spatial::PointSet;
use envisim_utils::utils::usize_to_f64;
pub use tc_parameters::{
    DbdConfiguration,
    TacticalConfigurationParameters,
};

use crate::{
    SamplingError,
    SamplingResult,
};

pub trait DistributionalDesigns {
    /// Construct a distributionally balanced design in a circular configuration
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    /// use envisim_utils::matrix::Matrix;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10)
    ///   .unwrap();
    /// let opts = SamplingOptions::new_equal(10, 2)?.set_spreading(m)?;
    /// let dbd_opts = DistributionalDesignOptions::default();
    /// let dbd = opts.dbd_circular(&mut rng, dbd_opts)?;
    /// let s: Vec<usize> = dbd.draw(&mut rng).collect();
    ///
    /// assert_eq!(s.len(), 2);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., & Prentius, W. (2026).
    /// Distributionally balanced sampling designs.
    /// arXiv preprint arXiv:2603.11916.
    /// <https://doi.org/10.48550/arXiv.2603.11916>
    fn dbd_circular<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<CircularConfiguration>
    where
        R: RandomNumberGenerator;
    /// Construct a distributionally balanced design in a circular configuration
    ///
    /// # Examples
    /// ```
    /// use envisim_samplr::*;
    /// use envisim_utils::random::*;
    /// use envisim_utils::matrix::Matrix;
    ///
    /// let mut rng = SmallRng::from_os_rng();
    /// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10)
    ///   .unwrap();
    /// let opts = SamplingOptions::new_equal(10, 2)?.set_spreading(m)?;
    /// let dbd_opts = DistributionalDesignOptions::default();
    /// let dbd = opts.dbd_tc(&mut rng, dbd_opts)?;
    /// let s: Vec<usize> = dbd.draw(&mut rng).collect();
    ///
    /// assert_eq!(s.len(), 2);
    /// # Ok::<(), SamplingError>(())
    /// ```
    ///
    /// # References
    /// Grafström, A., & Prentius, W. (2026).
    /// Distributionally balanced sampling designs via minimum tactical configurations.
    /// arXiv preprint arXiv:2603.24439.
    /// <https://doi.org/10.48550/arXiv.2603.24439>
    fn dbd_tc<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<TacticalConfiguration>
    where
        R: RandomNumberGenerator;
    /// Calculate mean and sd for the dbd circular design at specific iteration intervals
    fn dbd_circular_iterations<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: usize,
        by: usize,
    ) -> SamplingResult<Vec<f64>>
    where
        R: RandomNumberGenerator;
    /// Calculate mean and sd for the dbd tc design at specific iteration intervals
    fn dbd_tc_iterations<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: usize,
        by: usize,
    ) -> SamplingResult<Vec<f64>>
    where
        R: RandomNumberGenerator;
}

impl<SOP, M> DistributionalDesigns for SamplingOptions<'_, ProbabilitySpecEqual, SOP, M>
where
    SOP: PointSet<f64>,
{
    fn dbd_circular<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<CircularConfiguration>
    where
        R: RandomNumberGenerator,
    {
        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let max_iter = self.max_iterations();
        let eps = self.eps();
        let spreading_data = self.spreading()?.data();

        let mut v = match DbdCircular::new(&dbs_options, spreading_data, sample_size, eps) {
            Ok(v) => v,
            Err(c) => return Ok(c),
        };
        v.run(rng, max_iter);

        Ok(v.into_optimal_configuration())
    }
    fn dbd_tc<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
    ) -> SamplingResult<TacticalConfiguration>
    where
        R: RandomNumberGenerator,
    {
        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let max_iter = self.max_iterations();
        let eps = self.eps();
        let spreading_data = self.spreading()?.data();

        let mut v = match DbdTacticalConfiguration::new(
            rng,
            &dbs_options,
            spreading_data,
            sample_size,
            eps,
        ) {
            Ok(v) => v,
            Err(c) => return Ok(c),
        };
        v.run(rng, max_iter);

        Ok(v.into_optimal_configuration())
    }

    fn dbd_circular_iterations<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: usize,
        by: usize,
    ) -> SamplingResult<Vec<f64>>
    where
        R: RandomNumberGenerator,
    {
        if to < by || by == 0 {
            return Err(SamplingError::MaxIterations(unsafe {
                NonZeroUsize::new_unchecked(1)
            }));
        }

        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let sample_size_float = usize_to_f64(sample_size.get());
        let eps = self.eps();
        let spreading_data = self.spreading()?.data();

        let mut v = match DbdCircular::new(&dbs_options, spreading_data, sample_size, eps) {
            Ok(v) => v,
            Err(_) => {
                return Ok(vec![0.0; to / by * 2]);
            }
        };

        let mut res: Vec<f64> = Vec::with_capacity(to / by * 2);
        let by_nz = unsafe { NonZeroUsize::new_unchecked(by) };

        let mut iters = by;
        while iters <= to {
            v.run(rng, by_nz);

            let optimal_conf = v.optimal_configuration();
            let mean = optimal_conf.average_energy();
            let mut sd = 0.0;

            for sid in 0..optimal_conf.tcp().n_samples().get() {
                let energy = optimal_conf.nenergy_of_sample(v.ed(), sid) / sample_size_float;
                sd += (energy - mean).powi(2);
            }

            sd = (sd / usize_to_f64(optimal_conf.tcp().n_samples().get())).sqrt();
            res.push(mean);
            res.push(sd);
            iters += by;
        }

        Ok(res)
    }
    fn dbd_tc_iterations<R>(
        &self,
        rng: &mut R,
        dbs_options: DistributionalDesignOptions,
        to: usize,
        by: usize,
    ) -> SamplingResult<Vec<f64>>
    where
        R: RandomNumberGenerator,
    {
        if to < by || by == 0 {
            return Err(SamplingError::MaxIterations(unsafe {
                NonZeroUsize::new_unchecked(1)
            }));
        }

        let sample_size =
            NonZeroUsize::new(self.sample_size()).ok_or(SamplingError::ZeroSampleSize)?;
        let sample_size_float = usize_to_f64(sample_size.get());
        let eps = self.eps();
        let spreading_data = self.spreading()?.data();

        let mut v = match DbdTacticalConfiguration::new(
            rng,
            &dbs_options,
            spreading_data,
            sample_size,
            eps,
        ) {
            Ok(v) => v,
            Err(_) => {
                return Ok(vec![0.0; to / by * 2]);
            }
        };

        let mut res: Vec<f64> = Vec::with_capacity(to / by * 2);
        let n_buckets_float = usize_to_f64(v.tcp().n_samples().get());
        let by_nz = unsafe { NonZeroUsize::new_unchecked(by) };

        let mut iters = by;
        while iters <= to {
            v.run(rng, by_nz);

            let optimal_conf = v.optimal_configuration();
            let mean = optimal_conf.average_energy();
            let mut sd = 0.0;

            for sid in 0..optimal_conf.tcp().n_samples().get() {
                let energy = optimal_conf.nenergy_of_sample(v.ed(), sid) / sample_size_float;
                sd += (energy - mean).powi(2);
            }

            sd = (sd / n_buckets_float).sqrt();
            res.push(mean);
            res.push(sd);
            iters += by;
        }

        Ok(res)
    }
}
