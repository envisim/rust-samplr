// Copyright (C) 2025 Wilmer Prentius.
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

//! Provides an interface for drawing random numbers
#![allow(
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "used in almost every random number function"
)]

use std::ops::Range;
use std::slice::from_raw_parts_mut;

pub use rand_core::Rng;

use crate::number_traits::Number;

pub trait RngFloat: Rng {
    /// Generates the next uniform number in $[0.0, 1.0)$
    #[must_use]
    fn next_f64(&mut self) -> f64;
}

pub trait RandomNumber<R>: Number {
    /// Generates a random number of the type.
    #[must_use]
    fn rand(rng: &mut R) -> Self;
    /// Generates a random number in a `range`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[must_use]
    fn rand_in(rng: &mut R, range: Range<Self>) -> Self;
}
pub trait RandomNumberVec<R>: RandomNumber<R> {
    /// Fills `dest` with random values
    #[inline]
    fn rand_n(rng: &mut R, dest: &mut [Self]) {
        for v in dest.iter_mut() {
            *v = Self::rand(rng);
        }
    }
    /// Fills `dest` with random values in a `range`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[inline]
    fn rand_in_n(rng: &mut R, dest: &mut [Self], range: Range<Self>) {
        assert!(!range.is_empty(), "range is empty");
        for v in dest.iter_mut() {
            *v = Self::rand_in(rng, range.clone());
        }
    }
}

/// Macro for implementing `RandomNumber` for unsigned integers
macro_rules! random_number_impl_uint {
    ($t:ty,$rng:ident => $rand:expr) => {
        random_number_impl_uint!($t, $rng => $rand, vec => {});
    };
    ($t:ty,$rng:ident => $rand:expr, vec => {$($rnv_methods:item)*}) => {
        impl<R> RandomNumber<R> for $t
        where
            R: Rng,
        {
            #[inline]
            fn rand($rng: &mut R) -> Self { $rand }
            #[inline]
            fn rand_in($rng: &mut R, range: Range<Self>) -> Self {
                assert!(!range.is_empty());
                let b = range.end - range.start;
                let rem = ((Self::MAX % b) + 1) % b;
                let m = Self::MAX - rem;
                loop {
                    let u = Self::rand($rng);
                    if u <= m {
                        return range.start + (u % b);
                    }
                }
            }
        }
        impl<R> RandomNumberVec<R> for $t
        where
            R: Rng,
        {
            $($rnv_methods)*
        }
    };
}
/// Macro for implementing `RandomNumber` for signed integers
macro_rules! random_number_impl_sint {
    ($ts:ty,$tu:ty) => {
        random_number_impl_sint!($ts, $tu, vec => {});
    };
    ($ts:ty,$tu:ty, vec => {$($rnv_methods:item)*}) => {
        impl<R> RandomNumber<R> for $ts
        where
            R: Rng,
        {
            #[inline]
            fn rand(rng: &mut R) -> Self { <$tu>::rand(rng) as $ts }
            #[inline]
            fn rand_in(rng: &mut R, range: Range<Self>) -> Self {
                assert!(!range.is_empty());
                let end = range.end as $tu;
                let start = range.start as $tu;
                let w = end.wrapping_sub(start);
                range.start.wrapping_add(<$tu>::rand_in(rng, 0..w) as $ts)
            }
        }
        impl<R> RandomNumberVec<R> for $ts
        where
            R: Rng,
        {
            #[inline]
            fn rand_in_n(rng: &mut R, dest: &mut [Self], range: Range<Self>)  {
                assert!(!range.is_empty(), "range is empty");
                if dest.is_empty() {return;}
                let end = range.end as $tu;
                let start = range.start as $tu;
                let w = end.wrapping_sub(start);
                // SAFETY: $ts is the size of $tu
                let bytes = unsafe {
                    from_raw_parts_mut(
                        dest.as_mut_ptr().cast::<$tu>(),
                        dest.len()
                    )
                };
                <$tu>::rand_in_n(rng, bytes, 0..w);
                for v in dest.iter_mut() {
                    *v = v.wrapping_add(range.start);
                }
            }

            $($rnv_methods)*
        }
    };
}

random_number_impl_uint!(u8, rng => rng.next_u32() as u8, vec => {
    #[inline]
    fn rand_n(rng: &mut R, dest: &mut [Self]) {
        rng.fill_bytes(dest);
    }
    #[inline]
    fn rand_in_n(rng: &mut R, dest: &mut [Self], range: Range<Self>)  {
        assert!(!range.is_empty(), "range is empty");
        if dest.is_empty() {return;}

        let b = range.end - range.start;
        let rem = ((Self::MAX % b) + 1) % b;
        let m = Self::MAX - rem;

        let mut i = 0;
        while i < dest.len() {
            // Fill remaining with candidates
            rng.fill_bytes(&mut dest[i..]);
            let mut end = i;
            for j in i..dest.len() {
                if dest[j] <= m {
                    dest[end] = range.start + (dest[j] % b);
                    end += 1;
                }
            }
            i = end;
        }
    }
});
random_number_impl_uint!(u16, rng => rng.next_u32() as u16, vec => {
    #[inline]
    fn rand_n(rng: &mut R, dest: &mut [Self])  {
        // SAFETY: u8 is half the size of u16
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len() * 2
            )
        };
        rng.fill_bytes(bytes);
    }
    #[inline]
    fn rand_in_n(rng: &mut R, dest: &mut [Self], range: Range<Self>)  {
        assert!(!range.is_empty(), "range is empty");
        if dest.is_empty() {return;}

        let b = range.end - range.start;
        let rem = ((Self::MAX % b) + 1) % b;
        let m = Self::MAX - rem;

        let mut i = 0;
        while i < dest.len() {
            // Fill remaining with candidates
            Self::rand_n(rng, &mut dest[i..]);
            let mut end = i;
            for j in i..dest.len() {
                if dest[j] <= m {
                    dest[end] = range.start + (dest[j] % b);
                    end += 1;
                }
            }
            i = end;
        }
    }
});
random_number_impl_uint!(u32, rng => rng.next_u32());
random_number_impl_uint!(u64, rng => rng.next_u64());
random_number_impl_uint!(u128, rng => {
    let high = rng.next_u64() as u128;
    let low = rng.next_u64() as u128;
    (high << 64) | low
});
#[cfg(target_pointer_width = "32")]
random_number_impl_uint!(usize, rng => rng.next_u32() as usize);
#[cfg(target_pointer_width = "64")]
random_number_impl_uint!(usize, rng => rng.next_u64() as usize);

random_number_impl_sint!(i8, u8, vec => {
    #[inline]
    fn rand_n(rng: &mut R, dest: &mut [Self])  {
        // SAFETY: i8 is the size of u8
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len()
            )
        };
        rng.fill_bytes(bytes);
    }
});
random_number_impl_sint!(i16, u16, vec => {
    #[inline]
    fn rand_n(rng: &mut R, dest: &mut [Self])  {
        // SAFETY: u8 is half the size of i16
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len() * 2
            )
        };
        rng.fill_bytes(bytes);
    }
});
random_number_impl_sint!(i32, u32);
random_number_impl_sint!(i64, u64);
random_number_impl_sint!(i128, u128);
#[cfg(target_pointer_width = "32")]
random_number_impl_sint!(isize, usize);
#[cfg(target_pointer_width = "64")]
random_number_impl_sint!(isize, usize);

impl<R> RandomNumber<R> for f64
where
    R: RngFloat,
{
    #[inline]
    fn rand(rng: &mut R) -> Self { rng.next_f64() }
    #[inline]
    fn rand_in(rng: &mut R, range: Range<Self>) -> Self {
        assert!(!range.is_empty(), "range is empty");
        loop {
            let u = range.start + (range.end - range.start) * rng.next_f64();
            if u < range.end {
                return u;
            }
        }
    }
}
impl<R> RandomNumberVec<R> for f64 where R: RngFloat {}

/// Returns a random element of a slice, or `None` if the slice is empty.
#[must_use]
#[inline]
pub fn random_element<'bslice, R, T>(rng: &mut R, slice: &'bslice [T]) -> Option<&'bslice T>
where
    R: Rng,
{
    if slice.is_empty() {
        return None;
    }
    let index = usize::rand_in(rng, 0..slice.len());
    Some(&slice[index])
}

/// Returns `Some(true)` with probability `a / (a + b)` for two positive numbers, or `None` if all
/// numbers are non-positive.
#[must_use]
#[inline]
pub fn random_weighted<R, N>(rng: &mut R, a: N, b: N) -> Option<bool>
where
    R: RngFloat,
    N: RandomNumber<R>,
{
    let range = N::ZERO..(a + b);
    if !range.contains(&a) || !range.contains(&b) || range.is_empty() {
        return None;
    }
    Some(N::rand_in(rng, range) < b)
}

#[cfg(feature = "rand")]
mod small_rng {
    //! Implements [`RngFloat`] for [`rand::rngs::SmallRng`] if the feature `"rand"` is activated.

    pub use rand::SeedableRng;
    use rand::rngs::{
        SmallRng,
        SysRng,
    };
    use rand::{
        RngExt,
        TryRng,
    };

    use super::RngFloat;

    impl RngFloat for SmallRng {
        #[must_use]
        #[inline]
        fn next_f64(&mut self) -> f64 { self.random::<f64>() }
    }

    /// Tries to construct a [`SmallRng`] using [`SysRng`].
    ///
    /// # Errors
    /// Returns an error if rng cannot be constructed from [`SysRng`], see
    /// [`rand::rngs::SmallRng::try_from_rng`]
    #[inline]
    pub fn try_sys_rng() -> Result<SmallRng, <SysRng as TryRng>::Error> {
        SmallRng::try_from_rng(&mut SysRng)
    }
}

#[cfg(feature = "rand")]
pub use small_rng::*;
