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

//! Utils for using R randomness and RNG state in rust

use std::cmp::Ordering;

pub use envisim_utils::random::RandomNumberGenerator;

unsafe extern "C" {
    fn GetRNGstate();
    fn PutRNGstate();
    fn unif_rand() -> f64;
    // fn norm_rand() -> f64;
    // fn exp_rand() -> f64;
}

/// Holds the R RNG state
#[must_use]
pub struct RRng();

impl RRng {
    /// Gets the current RNG state from R
    #[inline]
    pub fn new() -> Self {
        // SAFETY: C-R interface, OK as long as we drop on destruct
        unsafe { GetRNGstate() };
        RRng()
    }
}

impl Drop for RRng {
    /// Puts the RNG state back to R
    #[inline]
    fn drop(&mut self) {
        // SAFETY: C-R interface, just putting back RNG state
        unsafe { PutRNGstate() };
    }
}

impl RandomNumberGenerator for RRng {
    #[inline]
    fn rf64(&mut self) -> f64 {
        // SAFETY: C-R interface
        unsafe { unif_rand() }
    }
    #[expect(
        clippy::cast_sign_loss,
        clippy::as_conversions,
        reason = "intended behaviour"
    )]
    #[inline]
    fn ru32(&mut self) -> u32 { self.ri32() as u32 }
    #[inline]
    fn ru32_to(&mut self, b: u32) -> u32 {
        loop {
            let u = self.ru32();
            let m = u32::MAX - (u32::MAX % b);
            if u < m {
                return u % b;
            }
        }
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "intended behaviour"
    )]
    #[inline]
    fn ri32(&mut self) -> i32 {
        let f = self.rf64() * 2.0 - 1.0;
        (f * f64::from(i32::MAX)).floor() as i32
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "intended behaviour"
    )]
    #[inline]
    fn ri32_in(&mut self, a: i32, b: i32) -> Option<i32> {
        self.rf64_in(a.into(), b.into()).map(|v| v.floor() as i32)
    }
    #[inline]
    fn ru64(&mut self) -> u64 {
        let high = u64::from(self.ru32());
        let low = u64::from(self.ru32());
        (high << 32) | low
    }
    #[inline]
    fn ru64_to(&mut self, b: u64) -> u64 {
        loop {
            let u = self.ru64();
            let m = u64::MAX - (u64::MAX % b);
            if u < m {
                return u % b;
            }
        }
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_wrap,
        reason = "intended behaviour"
    )]
    #[inline]
    fn ri64(&mut self) -> i64 { self.ru64() as i64 }
    #[expect(
        clippy::cast_sign_loss,
        clippy::as_conversions,
        reason = "intended behaviour"
    )]
    #[inline]
    fn ri64_in(&mut self, a: i64, b: i64) -> Option<i64> {
        match a.cmp(&b) {
            Ordering::Greater => return None,
            Ordering::Equal => return Some(a),
            Ordering::Less => (),
        };

        let diff = (b as u64).wrapping_sub(a as u64);
        // d can be 0...u32::MAX...i64::MAX / infty

        if let Ok(d) = u32::try_from(diff) {
            return Some(i64::from(self.ru32_to(d)) + a);
        }

        loop {
            let unif = self.ru64() & 0x7FFF_FFFF_FFFF_FFFF;
            let m_cap = u64::MAX - (u64::MAX) % diff;
            if unif < m_cap {
                #[expect(clippy::cast_possible_wrap, reason = "intended behaviour")]
                return Some((unif % diff) as i64 + a);
            }
        }
    }
}
