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

pub trait RandomNumberGenerator {
    /// Generates a uniform number in [`0.0`, `1.0`)
    #[must_use]
    fn rf64(&mut self) -> f64;
    /// Generates a uniform number in [`0.0`, `b`)
    #[must_use]
    #[inline]
    fn rf64_to(&mut self, b: f64) -> Option<f64> {
        if 0.0 < b {
            Some(self.rf64() * b)
        } else if b == 0.0 {
            Some(0.0)
        } else {
            None
        }
    }
    /// Generates a uniform number in [`a`, `b`)
    #[must_use]
    #[inline]
    fn rf64_in(&mut self, a: f64, b: f64) -> Option<f64> { self.rf64_to(b - a).map(|v| v + a) }

    /// Draws a random integer
    #[must_use]
    fn ru32(&mut self) -> u32;
    /// Draws a number from {`0`, ..., `u32::MAX-1`}
    #[must_use]
    fn ru32_to(&mut self, b: u32) -> u32;
    /// Draws a number from {`a`, ..., `b-1`}
    #[must_use]
    #[inline]
    fn ru32_in(&mut self, a: u32, b: u32) -> Option<u32> {
        (a < b).then_some(self.ru32_to(b - a) + a)
    }

    /// Draws a random integer
    #[expect(
        clippy::cast_possible_wrap,
        clippy::as_conversions,
        reason = "intended behaviour"
    )]
    #[must_use]
    #[inline]
    fn ri32(&mut self) -> i32 { self.ru32() as i32 }
    /// Draws a number from {`a`, ..., `b-1`}
    #[must_use]
    fn ri32_in(&mut self, a: i32, b: i32) -> Option<i32>;

    // Draws a random integer
    #[must_use]
    fn ru64(&mut self) -> u64;
    /// Draws a number from {`0`, ..., `b-1`}
    #[must_use]
    fn ru64_to(&mut self, b: u64) -> u64;
    /// Draws a number from {`a`, ..., `b-1`}
    #[must_use]
    #[inline]
    fn ru64_in(&mut self, a: u64, b: u64) -> Option<u64> {
        (a < b).then_some(self.ru64_to(b - a) + a)
    }

    /// Draws a random integer
    #[expect(
        clippy::cast_possible_wrap,
        clippy::as_conversions,
        reason = "intended behaviour"
    )]
    #[must_use]
    #[inline]
    fn ri64(&mut self) -> i64 { self.ru64() as i64 }
    /// Draws a number from {`a`, ..., `b-1`}
    #[must_use]
    fn ri64_in(&mut self, a: i64, b: i64) -> Option<i64>;

    // Draws a random integer
    #[cfg(target_pointer_width = "32")]
    #[expect(clippy::as_conversions, reason = "intended behaviour")]
    #[must_use]
    #[inline]
    fn rusize(&mut self) -> usize { self.ru32() as usize }
    /// Draws a number from {`0`, ..., `b-1`}
    #[cfg(target_pointer_width = "32")]
    #[expect(clippy::as_conversions, reason = "intended behaviour")]
    #[must_use]
    #[inline]
    fn rusize_to(&mut self, b: usize) -> usize { self.ru32_to(b as u32) as usize }
    /// Draws a random integer
    #[cfg(target_pointer_width = "64")]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "intended behaviour"
    )]
    #[must_use]
    #[inline]
    fn rusize(&mut self) -> usize { self.ru64() as usize }
    /// Draws a number from {`0`, ..., `b-1`}
    #[cfg(target_pointer_width = "64")]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "intended behaviour"
    )]
    #[must_use]
    #[inline]
    fn rusize_to(&mut self, b: usize) -> usize { self.ru64_to(b as u64) as usize }
    /// Draws a number from {`a`, ..., `b-1`}
    #[must_use]
    #[inline]
    fn rusize_in(&mut self, a: usize, b: usize) -> Option<usize> {
        (a < b).then_some(self.rusize_to(b - a) + a)
    }

    /// Returns `true` with probability `p`.
    #[expect(
        clippy::float_cmp,
        reason = "0.0 and 1.0 are special cases to be handled separately"
    )]
    #[must_use]
    #[inline]
    fn rbern(&mut self, p: f64) -> Option<bool> {
        if (0.0..=1.0).contains(&p) {
            if p == 0.0 {
                Some(false)
            } else if p == 1.0 {
                Some(true)
            } else {
                Some(self.rf64() < p)
            }
        } else {
            None
        }
    }
    /// Returns a random element of a vector
    #[must_use]
    #[inline]
    fn relement<'bslice, T>(&mut self, slice: &'bslice [T]) -> Option<&'bslice T> {
        (!slice.is_empty()).then_some(&slice[self.rusize_to(slice.len())])
    }
    /// Returns `Some(true)` with probability `p1 / (p0 + p1)` for two positive numbers.
    /// Returns `None` if any number is negative, or if both are zero.
    #[must_use]
    #[inline]
    fn one_of_f64(&mut self, p0: f64, p1: f64) -> Option<bool> {
        (0.0 <= p0 && 0.0 <= p1 && !(p0 == 0.0 && p1 == 0.0)).then_some(
            self.rf64_to(p0 + p1)
                .expect("sum of two positives to be positive")
                < p1,
        )
    }
}

#[cfg(feature = "rand")]
mod small_rng {
    //! Implements the [`RandomNumberGenerator`] for [`rand::rngs::SmallRng`] if the feature
    //! `"rand"` is activated.

    pub use rand::SeedableRng;
    use rand::rngs::{
        SmallRng as SRNG,
        SysRng,
    };
    use rand::{
        RngExt,
        TryRng,
    };

    /// let rng = StdRng::try_from_rng(&mut OsRng).unwrap();
    use super::RandomNumberGenerator;

    #[must_use]
    pub struct SmallRng(SRNG);

    impl SeedableRng for SmallRng {
        type Seed = [u8; 32];
        #[inline]
        fn from_seed(seed: Self::Seed) -> Self { SmallRng(SRNG::from_seed(seed)) }
        #[inline]
        fn seed_from_u64(state: u64) -> Self { SmallRng(SRNG::seed_from_u64(state)) }
    }
    impl SmallRng {
        pub fn try_sys_rng() -> Result<Self, <SysRng as TryRng>::Error> {
            SRNG::try_from_rng(&mut SysRng).map(|rng| SmallRng(rng))
        }
    }

    impl RandomNumberGenerator for SmallRng {
        #[inline]
        fn rf64(&mut self) -> f64 { self.0.random::<f64>() }
        #[inline]
        fn ru32(&mut self) -> u32 { self.0.random::<u32>() }
        #[inline]
        fn ru32_to(&mut self, b: u32) -> u32 { self.0.random_range(0..b) }
        #[inline]
        fn ru32_in(&mut self, a: u32, b: u32) -> Option<u32> {
            (a < b).then_some(self.0.random_range(a..b))
        }
        #[inline]
        fn ri32(&mut self) -> i32 { self.0.random::<i32>() }
        #[inline]
        fn ri32_in(&mut self, a: i32, b: i32) -> Option<i32> { Some(self.0.random_range(a..b)) }
        #[inline]
        fn ru64(&mut self) -> u64 { self.0.random::<u64>() }
        #[inline]
        fn ru64_to(&mut self, b: u64) -> u64 { self.0.random_range(0..b) }
        #[inline]
        fn ru64_in(&mut self, a: u64, b: u64) -> Option<u64> {
            (a < b).then_some(self.0.random_range(a..b))
        }
        #[inline]
        fn ri64(&mut self) -> i64 { self.0.random::<i64>() }
        #[inline]
        fn ri64_in(&mut self, a: i64, b: i64) -> Option<i64> { Some(self.0.random_range(a..b)) }
        #[inline]
        fn rusize(&mut self) -> usize { self.0.random_range(0..usize::MAX) }
        #[inline]
        fn rusize_to(&mut self, b: usize) -> usize { self.0.random_range(0..b) }
        #[inline]
        fn rusize_in(&mut self, a: usize, b: usize) -> Option<usize> {
            (a < b).then_some(self.0.random_range(a..b))
        }
    }

    impl From<SRNG> for SmallRng {
        #[inline]
        fn from(rng: SRNG) -> SmallRng { SmallRng(rng) }
    }

    #[must_use]
    pub struct SmallRngRef<'brng>(&'brng mut SRNG);

    impl RandomNumberGenerator for SmallRngRef<'_> {
        #[inline]
        fn rf64(&mut self) -> f64 { self.0.random::<f64>() }
        #[inline]
        fn ru32(&mut self) -> u32 { self.0.random::<u32>() }
        #[inline]
        fn ru32_to(&mut self, b: u32) -> u32 { self.0.random_range(0..b) }
        #[inline]
        fn ru32_in(&mut self, a: u32, b: u32) -> Option<u32> {
            (a < b).then_some(self.0.random_range(a..b))
        }
        #[inline]
        fn ri32(&mut self) -> i32 { self.0.random::<i32>() }
        #[inline]
        fn ri32_in(&mut self, a: i32, b: i32) -> Option<i32> { Some(self.0.random_range(a..b)) }
        #[inline]
        fn ru64(&mut self) -> u64 { self.0.random::<u64>() }
        #[inline]
        fn ru64_to(&mut self, b: u64) -> u64 { self.0.random_range(0..b) }
        #[inline]
        fn ru64_in(&mut self, a: u64, b: u64) -> Option<u64> {
            (a < b).then_some(self.0.random_range(a..b))
        }
        #[inline]
        fn ri64(&mut self) -> i64 { self.0.random::<i64>() }
        #[inline]
        fn ri64_in(&mut self, a: i64, b: i64) -> Option<i64> { Some(self.0.random_range(a..b)) }
        #[inline]
        fn rusize(&mut self) -> usize { self.0.random_range(0..usize::MAX) }
        #[inline]
        fn rusize_to(&mut self, b: usize) -> usize { self.0.random_range(0..b) }
        #[inline]
        fn rusize_in(&mut self, a: usize, b: usize) -> Option<usize> {
            (a < b).then_some(self.0.random_range(a..b))
        }
    }

    impl<'brng> From<&'brng mut SRNG> for SmallRngRef<'brng> {
        #[inline]
        fn from(rng: &mut SRNG) -> SmallRngRef { SmallRngRef(rng) }
    }
}

#[cfg(feature = "rand")]
pub use small_rng::*;
