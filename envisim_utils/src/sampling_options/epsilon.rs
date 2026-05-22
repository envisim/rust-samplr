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

//! Epsilon

use super::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::number_traits::Number;

#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct Epsilon<N = f64>(N);
impl<N> Epsilon<N> {
    pub fn get(self) -> N
    where
        N: Copy,
    {
        self.0
    }
    pub fn new(eps: N) -> SamplingOptionsResult<Self>
    where
        N: Number,
    {
        if !(N::ZERO..N::ONE).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        Ok(Self(eps))
    }
    pub fn is_zero(&self, value: N) -> bool
    where
        N: Number,
    {
        value.abs() <= self.0
    }
}

macro_rules! eps_impl_float {
    ($t:ty) => {
        impl Default for Epsilon<$t> {
            fn default() -> Self { Self(1e-12) }
        }
        impl TryFrom<$t> for Epsilon<$t> {
            type Error = SamplingOptionsError;
            fn try_from(value: $t) -> Result<Self, Self::Error> { Self::new(value) }
        }
    };
}
macro_rules! eps_impl_int {
    ($t:ty) => {
        impl Default for Epsilon<$t> {
            fn default() -> Self { Self(0) }
        }
    };
}

eps_impl_int!(usize);
eps_impl_int!(u8);
eps_impl_int!(u16);
eps_impl_int!(u32);
eps_impl_int!(u64);
eps_impl_int!(u128);

eps_impl_int!(isize);
eps_impl_int!(i8);
eps_impl_int!(i16);
eps_impl_int!(i32);
eps_impl_int!(i64);
eps_impl_int!(i128);

eps_impl_float!(f32);
eps_impl_float!(f64);
