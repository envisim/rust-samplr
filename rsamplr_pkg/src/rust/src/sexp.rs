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

//! Matrix utils

use std::slice::from_raw_parts;

use envisim_utils::matrix::MatrixBase;
use envisim_utils::sampling_options::{
    SamplingOptions,
    SpreadingOptions,
    UnequalProbabilities,
    UnequalProbabilitiesReal,
};
use envisim_utils::utils::SliceView;
use savvy::{
    RealSexp,
    savvy_err,
};
use savvy_ffi::REAL;

use crate::utils::{
    to_nzusize,
    to_usize,
};

/// Wrapper for `RealSexp`
pub struct RealSexpFatPtr {
    /// Underlying reference to data
    #[expect(dead_code, reason = "structure should own data")]
    sexp: RealSexp,
    /// Pointer to data
    ptr: *const f64,
    /// Length of data
    len: usize,
}

impl RealSexpFatPtr {
    /// Constructs the fat pointer from a `RealSexp`
    pub fn from_sexp(sexp: RealSexp) -> savvy::Result<Self> {
        if sexp.is_empty() {
            return Err(savvy_err!("sexp is empty"));
        }

        Ok(Self {
            // SAFETY:
            // Retrieving a pointer to the underlying data is mimicing savvy::RealSexp::as_slice()
            // behaviour.
            ptr: unsafe { REAL(sexp.0) },
            len: sexp.len(),
            sexp,
        })
    }

    /// Constructs a `RMatrix` from `RealSexp`
    #[inline]
    pub fn to_matrix(sexp: RealSexp) -> savvy::Result<RMatrix> {
        let rows = match sexp
            .get_dim()
            .ok_or_else(|| savvy_err!("object have no dimensions"))?
        {
            [_, _, _, ..] => Err(savvy_err!("object have too many dimensions")),
            [r, ..] => to_usize(*r),
            _ => Err(savvy_err!("object have no dimensions")),
        }?;

        Ok(MatrixBase::new(sexp.try_into()?, rows).expect("rows to be NonZeroUsize"))
    }
    /// Constructs spreading options from `RealSexp`
    #[inline]
    pub fn to_spreading_options<BSZ>(
        sexp: RealSexp,
        bucket_size: BSZ,
    ) -> savvy::Result<SpreadingOptions<RMatrix>>
    where
        BSZ: Into<Option<i32>>,
    {
        let data = Self::to_matrix(sexp)?;
        let mut opts = SpreadingOptions::new(data);
        if let Some(bucket_size) = bucket_size.into() {
            let bucket_size = to_nzusize(bucket_size)?;
            opts = opts.set_bucket_size(bucket_size)?;
        }
        Ok(opts)
    }
    /// Constructs a `ProbabilitySpec` from  `RealSexp`
    #[inline]
    pub fn to_probs_unequal(sexp: RealSexp) -> savvy::Result<RUnequalProbabilities> {
        Ok(RUnequalProbabilities::new(sexp.try_into()?)?)
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
        let probs = Self::to_probs_unequal(sexp)?;
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

impl TryFrom<RealSexp> for RealSexpFatPtr {
    type Error = savvy::Error;
    #[inline]
    fn try_from(sexp: RealSexp) -> Result<Self, Self::Error> { Self::from_sexp(sexp) }
}

impl SliceView for RealSexpFatPtr {
    type Elem = f64;
    #[inline]
    fn data(&self) -> &[Self::Elem] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY:
        // Reconstructing the slice from the fat pointer is mimicing savvy::RealSexp::as_slice()
        // behaviour.
        unsafe { from_raw_parts(self.ptr, self.len) }
    }
}

/// Type alias for `Matrix` using `RMatrixData`.
pub type RMatrix = MatrixBase<RealSexpFatPtr>;

/// Type alias for `UnequalProbabilitiesReal` using `RProbabilitiesData`
pub type RUnequalProbabilities = UnequalProbabilities<UnequalProbabilitiesReal<RealSexpFatPtr>>;
