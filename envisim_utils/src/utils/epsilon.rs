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

use std::fmt::{
    Display,
    Formatter,
    Result as FmtResult,
};

use super::Number;
use crate::sampling_options::{
    SamplingOptionsError,
    SamplingOptionsResult,
};

/// An epsilon-like value, used for comparisons between `N` representations
///
/// For ints, Epsilon is constricted to 0, and for floats Epsilon should be sufficiently small.
/// The constructor checks so that epsilon < 1.0, but this is almost guaranteed to be a too loose
/// restriction. The default epsilon is `1e-12`.
#[must_use]
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct Epsilon<N = f64>(N);

impl<N> Epsilon<N> {
    /// Gets the internal epsilon value
    #[must_use]
    #[inline]
    pub fn get(self) -> N
    where
        N: Copy,
    {
        self.0
    }
    /// Constructs a new `Epsilon`
    /// # Errors
    /// Returns an error if `eps` is not sufficiently small.
    #[inline]
    pub fn new(eps: N) -> SamplingOptionsResult<Self>
    where
        N: Number,
    {
        if !(N::ZERO..N::ONE).contains(&eps) {
            return Err(SamplingOptionsError::InvalidEpsilon);
        }
        Ok(Self(eps))
    }
    /// Returns `true` if the absolute `value` is almost zero
    #[must_use]
    #[inline]
    pub fn is_zero(&self, value: N) -> bool
    where
        N: Number,
    {
        value.abs() <= self.0
    }
    /// Returns `true` if the absolute difference between `a` and `b` is almost zero.
    #[must_use]
    #[inline]
    pub fn difference_is_zero(&self, a: N, b: N) -> bool
    where
        N: Number,
    {
        Number::abs_difference(a, b) <= self.0
    }
}

impl<N> Default for Epsilon<N>
where
    N: Number,
{
    #[inline]
    fn default() -> Self { Self(N::DEFAULT_EPSILON_VALUE) }
}

impl<N> Display for Epsilon<N>
where
    N: Number,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> FmtResult { write!(f, "{}", self.0) }
}

/// Implements epsilon for floats
macro_rules! eps_impl_float {
    ($t:ty) => {
        impl TryFrom<$t> for Epsilon<$t> {
            type Error = SamplingOptionsError;
            #[inline]
            fn try_from(value: $t) -> Result<Self, Self::Error> { Self::new(value) }
        }
    };
}
///// Implements epsilon for ints
// macro_rules! eps_impl_int {
//     ($t:ty) => {};
// }

// eps_impl_int!(usize);
// eps_impl_int!(u8);
// eps_impl_int!(u16);
// eps_impl_int!(u32);
// eps_impl_int!(u64);
// eps_impl_int!(u128);

// eps_impl_int!(isize);
// eps_impl_int!(i8);
// eps_impl_int!(i16);
// eps_impl_int!(i32);
// eps_impl_int!(i64);
// eps_impl_int!(i128);

eps_impl_float!(f32);
eps_impl_float!(f64);
