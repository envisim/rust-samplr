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

//! Utilites

use std::num::NonZeroUsize;

use num_traits::ToPrimitive;
use savvy::{
    OwnedIntegerSexp,
    Sexp,
    savvy_err,
};

/// Converts to usize
/// # Errors
/// If not possible to convert to usize
#[expect(clippy::needless_pass_by_value, reason = "can be reference")]
#[inline]
pub fn to_usize<T>(v: T) -> savvy::Result<usize>
where
    T: ToPrimitive,
{
    v.to_usize()
        .ok_or_else(|| savvy_err!("value must be non-negative"))
}

/// Converts to nonzerousize
/// # Errors
/// If not possible to convert to nonzerousize
#[expect(clippy::needless_pass_by_value, reason = "can be reference")]
#[inline]
pub fn to_nzusize<T>(v: T) -> savvy::Result<NonZeroUsize>
where
    T: ToPrimitive,
{
    v.to_usize()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| savvy_err!("value must be positive"))
}

/// Converts to i32
/// # Errors
/// If not possible to convert to i32
#[expect(clippy::needless_pass_by_value, reason = "can be reference")]
#[inline]
pub fn to_i32<T>(v: T) -> savvy::Result<i32>
where
    T: ToPrimitive,
{
    v.to_i32()
        .ok_or_else(|| savvy_err!("cannot convert to i32"))
}

/// Converts sample to [`Sexp`]
/// # Errors
/// If not possible to convert a sample index to i32
#[inline]
pub fn return_sample<T>(sample: T) -> savvy::Result<Sexp>
where
    T: AsRef<[usize]>,
{
    let slice = sample.as_ref();
    let mut out = OwnedIntegerSexp::new(slice.len())?;
    for (o, s) in out.iter_mut().zip(slice.iter()) {
        *o = to_i32(*s)? + 1;
    }
    Ok(Sexp::from(out))
}
