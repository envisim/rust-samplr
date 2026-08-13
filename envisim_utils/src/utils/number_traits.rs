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

//! Provides a trait [`Number`], extending `num_traits`

use std::cmp::Ordering;
use std::fmt::{
    Debug,
    Display,
};
use std::iter::Sum;

use num_integer::Integer;
use num_traits::{
    ConstOne,
    ConstZero,
    Float,
    NumAssign,
    NumCast,
};

use super::Epsilon;

/// An extension of [`num_traits::NumAssign`]
pub trait Number:
    Sized
    + Copy
    + PartialOrd
    + PartialEq
    + NumAssign
    + NumCast
    + ConstZero
    + ConstOne
    + Debug
    + Display
    + Sum
{
    /// The default epsilon value for the type, i.e. a small value to use in comparisons
    const DEFAULT_EPSILON_VALUE: Self;
    /// Returns the machine epsilon value, if possible for the type, otherwise zero.
    #[must_use]
    #[inline]
    fn machine_epsilon() -> Self { Self::ZERO }
    /// Returns the maximum value of the type
    #[must_use]
    fn max_value() -> Self;
    /// Returns true if `self` is finite
    #[must_use]
    #[inline]
    fn is_finite(self) -> bool { true }
    /// Returns true if `self` is positive and finite
    #[must_use]
    #[inline]
    fn is_pos_finite(self) -> bool { Self::ZERO < self }
    /// Returns the absolute value of `self`
    #[must_use]
    #[inline]
    fn abs(self) -> Self { self }
    ///Returns the absolute differencet between `self` and `other`.
    #[must_use]
    fn abs_difference(self, other: Self) -> Self;
    /// Returns the midpoint between `self` and `other`
    #[must_use]
    fn mid(self, other: Self) -> Self;
    /// Returns the ordering between `self` and `other`
    #[must_use]
    fn compare(&self, other: &Self) -> Ordering;
    /// Returns the [`DEFAULT_EPSILON_VALUE`] from the type of `self`
    #[inline]
    fn default_epsilon(&self) -> Epsilon<Self> { Epsilon::default() }
}
/// Floating point numbers
pub trait NumberFloat: Number + Float {}
impl<N> NumberFloat for N where N: Number + Float {}
/// Integer numbers
pub trait NumberInt: Number + Integer {}
impl<N> NumberInt for N where N: Number + Integer {}

/// Interanal macro that implements `Number` for floats
macro_rules! number_impl_float {
    ($t:ty) => {
        impl Number for $t {
            const DEFAULT_EPSILON_VALUE: Self = 1e-12;

            #[inline]
            fn machine_epsilon() -> Self { <$t>::EPSILON }
            #[inline]
            fn max_value() -> Self { <$t>::MAX }
            #[inline]
            fn is_finite(self) -> bool { <$t>::is_finite(self) }
            #[inline]
            fn is_pos_finite(self) -> bool { <$t>::ZERO < self && <$t>::is_finite(self) }
            #[inline]
            fn abs(self) -> Self { <$t>::abs(self) }
            #[inline]
            fn abs_difference(self, other: Self) -> Self { (self - other).abs() }
            #[inline]
            fn mid(self, other: Self) -> Self { <$t>::midpoint(self, other) }
            #[inline]
            fn compare(&self, other: &Self) -> Ordering {
                match (self.is_nan(), other.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => self.partial_cmp(other).unwrap(),
                }
            }
        }
    };
}
/// Interanal macro that implements `Number` for unsigned integers
macro_rules! number_impl_uint {
    ($t:ty) => {
        impl Number for $t {
            const DEFAULT_EPSILON_VALUE: Self = 0;

            #[inline]
            fn max_value() -> Self { <$t>::MAX }
            #[inline]
            fn abs_difference(self, other: Self) -> Self { <$t>::abs_diff(self, other) }
            #[inline]
            fn mid(self, other: Self) -> Self { <$t>::midpoint(self, other) }
            #[inline]
            fn compare(&self, other: &Self) -> Ordering { <$t>::cmp(self, other) }
        }
    };
}
/// Interanal macro that implements `Number` for signed integers
macro_rules! number_impl_sint {
    ($t:ty) => {
        impl Number for $t {
            const DEFAULT_EPSILON_VALUE: Self = 0;

            #[inline]
            fn max_value() -> Self { <$t>::MAX }
            #[inline]
            fn abs(self) -> Self { <$t>::abs(self) }
            #[inline]
            fn abs_difference(self, other: Self) -> Self { (self - other).abs() }
            #[inline]
            fn mid(self, other: Self) -> Self {
                // unstable until 1.87
                // <$t>::midpoint(self, other)
                // use rust impl directly, see
                // https://doc.rust-lang.org/src/core/num/mod.rs.html#184
                let t = ((self ^ other) >> 1) + (self & other);
                t + (if t < 0 { 1 } else { 0 } & (self ^ other))
            }
            #[inline]
            fn compare(&self, other: &Self) -> Ordering { <$t>::cmp(self, other) }
        }
    };
}

number_impl_uint!(usize);
number_impl_uint!(u8);
number_impl_uint!(u16);
number_impl_uint!(u32);
number_impl_uint!(u64);
number_impl_uint!(u128);

number_impl_sint!(isize);
number_impl_sint!(i8);
number_impl_sint!(i16);
number_impl_sint!(i32);
number_impl_sint!(i64);
number_impl_sint!(i128);

number_impl_float!(f32);
number_impl_float!(f64);
