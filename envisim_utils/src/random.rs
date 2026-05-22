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

pub use rand_core::{
    Rng,
    TryRng,
};

use crate::number_traits::Number;

pub trait FloatRng: Rng {
    /// Generates the next uniform number in $[0.0, 1.0)$
    #[must_use]
    fn next_f64(&mut self) -> f64;
}

pub trait Rand<N>: Rng
where
    N: Number,
{
    /// Generates a random number.
    #[must_use]
    fn rand(&mut self) -> N;
    /// Generates a random number in a `range`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[must_use]
    fn rand_in(&mut self, range: Range<N>) -> N;
    /// Generates a random number in a range `0..max`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[must_use]
    #[inline]
    fn rand_to(&mut self, max: N) -> N { self.rand_in(N::ZERO..max) }
}
pub trait RandSlice<N>: Rand<N>
where
    N: Number,
{
    /// Fills `dest` with random values
    #[inline]
    fn rand_n(&mut self, dest: &mut [N]) {
        for v in dest.iter_mut() {
            *v = self.rand();
        }
    }
    /// Fills `dest` with random values in a `range`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[inline]
    fn rand_in_n(&mut self, dest: &mut [N], range: Range<N>) {
        assert!(!range.is_empty(), "range is empty");
        for v in dest.iter_mut() {
            *v = self.rand_in(range.clone());
        }
    }
    /// Fills `dest` with random values in a range `0..max`.
    ///
    /// # Panics
    /// Panics if range is empty.
    #[inline]
    fn rand_to_n(&mut self, dest: &mut [N], max: N) { self.rand_in_n(dest, N::ZERO..max) }
}

/// Macro for implementing `RandomNumber` for unsigned integers
macro_rules! rand_impl_uint {
    ($t:ty, $self:ident => $body:expr) => {
        rand_impl_uint!($t, $self => $body, vec => {});
    };
    ($t:ty, $self:ident => $body:expr, vec => {$($rnv_methods:item)*}) => {
        impl<R> Rand<$t> for R
        where
            R: Rng,
        {
            #[inline]
            fn rand(&mut $self) -> $t { $body }
            #[inline]
            fn rand_in(&mut self, range: Range<$t>) -> $t {
                assert!(!range.is_empty());
                let b = range.end - range.start;
                let rem = ((<$t>::MAX % b) + 1) % b;
                let m = <$t>::MAX - rem;
                loop {
                    let u: $t = self.rand();
                    if u <= m {
                        return range.start + (u % b);
                    }
                }
            }
        }
        impl<R> RandSlice<$t> for R
        where
            R: Rng,
        {
            $($rnv_methods)*
        }
    };
}
/// Macro for implementing `RandomNumber` for signed integers
macro_rules! rand_impl_sint {
    ($ts:ty, $tu:ty) => {
        rand_impl_sint!($ts, $tu, vec => {});
    };
    ($ts:ty, $tu:ty, vec => {$($rnv_methods:item)*}) => {
        impl<R> Rand<$ts> for R
        where
            R: Rng,
        {
            #[inline]
            fn rand(&mut self) -> $ts { <Self as Rand<$tu>>::rand(self) as $ts }
            #[inline]
            fn rand_in(&mut self, range: Range<$ts>) -> $ts {
                assert!(!range.is_empty());
                let end = range.end as $tu;
                let start = range.start as $tu;
                let w = end.wrapping_sub(start);
                range.start.wrapping_add(<Self as Rand<$tu>>::rand_in(self, 0..w) as $ts)
            }
        }
        impl<R> RandSlice<$ts> for R
        where
            R: Rng,
        {
            #[inline]
            fn rand_in_n(&mut self, dest: &mut [$ts], range: Range<$ts>)  {
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
                self.rand_in_n(bytes, 0..w);
                for v in dest.iter_mut() {
                    *v = v.wrapping_add(range.start);
                }
            }

            $($rnv_methods)*
        }
    };
}

rand_impl_uint!(u8, self => self.next_u32() as u8, vec => {
    #[inline]
    fn rand_n(&mut self, dest: &mut [u8]) {
        self.fill_bytes(dest);
    }
    #[inline]
    fn rand_in_n(&mut self, dest: &mut [u8], range: Range<u8>)  {
        assert!(!range.is_empty(), "range is empty");
        if dest.is_empty() {return;}

        let b = range.end - range.start;
        let rem = ((u8::MAX % b) + 1) % b;
        let m = u8::MAX - rem;

        let mut i = 0;
        while i < dest.len() {
            // Fill remaining with candidates
            self.fill_bytes(&mut dest[i..]);
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
rand_impl_uint!(u16, self => self.next_u32() as u16, vec => {
    #[inline]
    fn rand_n(&mut self, dest: &mut [u16])  {
        // SAFETY: u8 is half the size of u16
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len() * 2
            )
        };
        self.fill_bytes(bytes);
    }
    #[inline]
    fn rand_in_n( &mut self, dest: &mut [u16], range: Range<u16>)  {
        assert!(!range.is_empty(), "range is empty");
        if dest.is_empty() {return;}

        let b = range.end - range.start;
        let rem = ((u16::MAX % b) + 1) % b;
        let m = u16::MAX - rem;

        let mut i = 0;
        while i < dest.len() {
            // Fill remaining with candidates
            self.rand_n(&mut dest[i..]);
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
rand_impl_uint!(u32, self => self.next_u32());
rand_impl_uint!(u64, self => self.next_u64());
rand_impl_uint!(u128, self => {
    let high = self.next_u64() as u128;
    let low = self.next_u64() as u128;
    (high << 64) | low
});
#[cfg(target_pointer_width = "32")]
rand_impl_uint!(usize, self => self.next_u32() as usize);
#[cfg(target_pointer_width = "64")]
rand_impl_uint!(usize, self => self.next_u64() as usize);

rand_impl_sint!(i8, u8, vec => {
    #[inline]
    fn rand_n(&mut self, dest: &mut [i8])  {
        // SAFETY: i8 is the size of u8
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len()
            )
        };
        self.fill_bytes(bytes);
    }
});
rand_impl_sint!(i16, u16, vec => {
    #[inline]
    fn rand_n(&mut self, dest: &mut [i16])  {
        // SAFETY: u8 is half the size of i16
        let bytes = unsafe {
            from_raw_parts_mut(
                dest.as_mut_ptr().cast::<u8>(),
                dest.len() * 2
            )
        };
        self.fill_bytes(bytes);
    }
});
rand_impl_sint!(i32, u32);
rand_impl_sint!(i64, u64);
rand_impl_sint!(i128, u128);
#[cfg(target_pointer_width = "32")]
rand_impl_sint!(isize, usize);
#[cfg(target_pointer_width = "64")]
rand_impl_sint!(isize, usize);

impl<R> Rand<f64> for R
where
    R: FloatRng,
{
    #[inline]
    fn rand(&mut self) -> f64 { self.next_f64() }
    #[inline]
    fn rand_in(&mut self, range: Range<f64>) -> f64 {
        assert!(!range.is_empty(), "range is empty");
        loop {
            let u = range.start + (range.end - range.start) * self.next_f64();
            if u < range.end {
                return u;
            }
        }
    }
}
impl<R> RandSlice<f64> for R where R: FloatRng {}

/// Returns a random element of a slice, or `None` if the slice is empty.
#[must_use]
#[inline]
pub fn random_element<'bslice, R, T>(rng: &mut R, slice: &'bslice [T]) -> Option<&'bslice T>
where
    R: Rand<usize>,
{
    if slice.is_empty() {
        return None;
    }
    let index: usize = rng.rand_in(0..slice.len());
    Some(&slice[index])
}

/// Returns `Some(true)` with probability `a / (a + b)` for two positive numbers, or `None` if all
/// numbers are non-positive.
#[must_use]
#[inline]
pub fn random_weighted<R, N>(rng: &mut R, a: N, b: N) -> Option<bool>
where
    R: Rand<N>,
    N: Number,
{
    let range = N::ZERO..(a + b);
    if !range.contains(&a) || !range.contains(&b) || range.is_empty() {
        return None;
    }
    Some(rng.rand_in(range) < b)
}

#[cfg(feature = "rand")]
mod small_rng {
    //! Implements [`RngFloat`] for [`rand::rngs::SmallRng`] if the feature `"rand"` is activated.

    pub use rand::SeedableRng;
    pub use rand::rngs::SmallRng;
    use rand::rngs::SysRng;
    use rand::{
        RngExt,
        TryRng,
    };

    use super::FloatRng;

    impl FloatRng for SmallRng {
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
