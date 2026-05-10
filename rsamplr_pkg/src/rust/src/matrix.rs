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

use std::num::NonZeroUsize;

pub use envisim_utils::matrix::Dimensions;
use envisim_utils::matrix::MatrixRef;
use savvy::{
    RealSexp,
    savvy_err,
};

/// Returns the nrow attribute from a [`RealSexp`].
/// # Panics
/// Panics if Sexp is not a matrix, or the dimension is 0
#[inline]
pub fn get_nrow(mat: &RealSexp) -> savvy::Result<NonZeroUsize> {
    let rows: usize = mat.get_dim().ok_or(savvy_err!("object is not matrix"))?[0]
        .try_into()
        .map_err(|_| savvy_err!("dimension must be positive"))?;
    NonZeroUsize::new(rows).ok_or(savvy_err!("dimension must be positive"))
}

/// Converts into a matrix
/// # Panics
/// Panics if the dimensions are invalid
#[inline]
pub fn to_matrix(mat: &[f64], nrow: NonZeroUsize) -> MatrixRef<'_, f64> {
    MatrixRef::new(mat, nrow).expect("matrix to have valid dimensions")
}
