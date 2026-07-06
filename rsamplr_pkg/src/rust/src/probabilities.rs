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

//! Probabilities

use envisim_utils::sampling_options::{
    SamplingOptions,
    UnequalProbabilities,
    UnequalProbabilitiesReal,
};
use envisim_utils::utils::SliceView;
use savvy::RealSexp;

use crate::utils::to_nzusize;

/// Wrapper for probability data
pub struct RProbabilitiesData(RealSexp);

/// Type alias for `UnequalProbabilitiesReal` using `RProbabilitiesData`
pub type RUnequalProbabilities = UnequalProbabilities<UnequalProbabilitiesReal<RProbabilitiesData>>;

impl SliceView for RProbabilitiesData {
    type Elem = f64;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.0.as_slice() }
}

impl RProbabilitiesData {
    /// Constructs a `ProbabilitySpec` from  `RealSexp`
    #[inline]
    pub fn to_unequal(sexp: RealSexp) -> savvy::Result<RUnequalProbabilities> {
        Ok(RUnequalProbabilities::new(RProbabilitiesData(sexp))?)
    }
    /// Constructs a `SamplingOptions` from  `RealSexp`
    #[inline]
    pub fn to_sampling_options<EPS, MAX>(
        sexp: RealSexp,
        eps: EPS,
        max_iter: MAX,
    ) -> savvy::Result<SamplingOptions<RUnequalProbabilities, (), ()>>
    where
        EPS: Into<Option<f64>>,
        MAX: Into<Option<i32>>,
    {
        let probs = Self::to_unequal(sexp)?;
        let mut opts = SamplingOptions::with_spec(probs);
        if let Some(eps) = eps.into() {
            opts = opts.set_eps(eps)?;
        }
        if let Some(max_iter) = max_iter.into() {
            let max_iter = to_nzusize(max_iter)?;
            opts = opts.set_max_iterations(max_iter)?;
        }
        Ok(opts)
    }
}
