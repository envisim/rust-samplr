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

use std::cmp::Ordering;
use std::fmt::Display;

use num_traits::{
    ConstOne,
    ConstZero,
    Float,
    NumAssign,
    NumCast,
};

pub trait Number:
    Sized + Copy + PartialOrd + PartialEq + NumAssign + NumCast + ConstZero + ConstOne + Display
{
    #[must_use]
    #[inline]
    fn epsilonish() -> Self { Self::ZERO }
    #[must_use]
    fn max_value() -> Self;

    #[must_use]
    #[inline]
    fn is_finite(self) -> bool { true }
    #[must_use]
    #[inline]
    fn is_pos_finite(self) -> bool { Self::ZERO < self }

    #[must_use]
    #[inline]
    fn abs(self) -> Self { self }
    #[must_use]
    fn abs_difference(self, other: Self) -> Self;
    #[must_use]
    fn mid(self, other: Self) -> Self;

    #[must_use]
    fn compare(&self, other: &Self) -> Ordering;
}
pub trait NumberFloat: Number + Float {}

#[cfg(any(test, feature = "test-utils"))]
pub trait NumberTest: Number + Display {
    const TEST_EPS: Self;
    #[must_use]
    #[inline]
    fn test_eps(&self) -> Self { Self::TEST_EPS }
    #[must_use]
    #[inline]
    fn approx_eq(self, other: Self) -> bool { Number::abs(self - other) <= Self::TEST_EPS }
    #[must_use]
    #[inline]
    fn approx_eq_eps(self, other: Self, eps: Self) -> bool {
        Number::abs_difference(self, other) <= eps
    }
}

/// Interanal macro that implements `Number` for floats
macro_rules! number_impl_float {
    ($t:ty) => {
        impl Number for $t {
            #[inline]
            fn epsilonish() -> Self { <$t>::EPSILON }
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

        impl NumberFloat for $t {}

        #[cfg(any(test, feature = "test-utils"))]
        impl NumberTest for $t {
            const TEST_EPS: Self = 1e-12;
        }
    };
}
/// Interanal macro that implements `Number` for unsigned integers
macro_rules! number_impl_uint {
    ($t:ty) => {
        impl Number for $t {
            #[inline]
            fn max_value() -> Self { <$t>::MAX }
            #[inline]
            fn abs_difference(self, other: Self) -> Self { <$t>::abs_diff(self, other) }
            #[inline]
            fn mid(self, other: Self) -> Self { <$t>::midpoint(self, other) }
            #[inline]
            fn compare(&self, other: &Self) -> Ordering { <$t>::cmp(self, other) }
        }

        #[cfg(any(test, feature = "test-utils"))]
        impl NumberTest for $t {
            const TEST_EPS: Self = 0;
            #[inline]
            fn approx_eq(self, other: Self) -> bool { self == other }
        }
    };
}
/// Interanal macro that implements `Number` for signed integers
macro_rules! number_impl_sint {
    ($t:ty) => {
        impl Number for $t {
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

        #[cfg(any(test, feature = "test-utils"))]
        impl NumberTest for $t {
            const TEST_EPS: Self = 0;
            #[inline]
            fn approx_eq(self, other: Self) -> bool { self == other }
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
