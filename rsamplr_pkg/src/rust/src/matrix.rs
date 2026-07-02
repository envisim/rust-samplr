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

use envisim_samplr::SpreadingOptions;
use envisim_utils::matrix::{
    MatrixBase,
    RawData,
};
use savvy::{
    RealSexp,
    savvy_err,
};

use crate::utils::{
    to_nzusize,
    to_usize,
};

/// Wrapper for matrix data
pub struct RMatrixData(RealSexp);
/// Type alias for `Matrix` using
pub type RMatrix = MatrixBase<RMatrixData>;
impl RawData for RMatrixData {
    type Elem = f64;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.0.as_slice() }
}
impl RMatrixData {
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
        Ok(MatrixBase::new(RMatrixData(sexp), rows).expect("rows to be NonZeroUsize"))
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
}
