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

//! Systematic sampling designs.
//!
//! Implements [`SystematicSampling`] for [`SamplingOptions`].

use envisim_utils::random::{
    Rand,
    Rng,
};
pub use envisim_utils::sampling_options::SamplingOptions;
use envisim_utils::sampling_options::{
    BaseProbabilitiesSpec,
    SamplingOptionsRng,
};
use num_traits::ConstZero;

pub use crate::error::SamplingError;
use crate::utils::shuffle;

/// Draws a systematic sample from the provided order.
#[must_use]
#[inline]
fn from_order<R, PS>(rng: &mut R, options: &PS, order: Vec<PS::Id>) -> Vec<PS::Id>
where
    R: Rand<PS::Value>,
    PS: BaseProbabilitiesSpec,
{
    let pmax = options.max();
    let mut sample = Vec::<PS::Id>::with_capacity(options.sample_size());
    let mut r = rng.rand_to(pmax);
    let mut psum = PS::Value::ZERO;

    for id in order {
        let pnext = psum + options.get(id).copied().expect("id to exist");
        if psum <= r && r < pnext {
            sample.push(id);
            r += pmax;
        }

        psum = pnext;
    }

    sample
}

/// Provides systematic sampling methods.
pub trait SystematicSampling<ID, R>
where
    R: Rng,
{
    /// Draw a systematic sample, using the provided order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.systematic(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic(&self, rng: &mut R) -> Vec<ID>;
    /// Draw a systematic sample, using a random order.
    ///
    /// # Examples
    /// ```
    /// # use envisim_samplr::*;
    /// # use envisim_utils::random::*;
    /// let mut rng = try_sys_rng().unwrap();
    /// let p: Vec<f64> = vec![0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
    /// let s = SamplingOptions::new(p)?.systematic_random_order(&mut rng);
    /// assert_eq!(s.len(), 5);
    /// # Ok::<(), SamplingError>(())
    /// ```
    fn systematic_random_order(&self, rng: &mut R) -> Vec<ID>;
}
impl<R, PS, AUX> SystematicSampling<PS::Id, R> for SamplingOptions<PS, AUX>
where
    PS: BaseProbabilitiesSpec,
    R: SamplingOptionsRng<PS>,
{
    #[inline]
    fn systematic(&self, rng: &mut R) -> Vec<PS::Id> {
        let order: Vec<PS::Id> = self.probabilities().ids().collect();
        from_order(rng, self.probabilities(), order)
    }
    #[inline]
    fn systematic_random_order(&self, rng: &mut R) -> Vec<PS::Id> {
        let order = shuffle(rng, self.probabilities().ids().collect());
        from_order(rng, self.probabilities(), order)
    }
}
