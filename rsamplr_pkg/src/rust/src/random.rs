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
#![allow(
    clippy::as_conversions,
    clippy::little_endian_bytes,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "used in almost every random number function"
)]

use std::convert::Infallible;

use envisim_utils::random::{
    FloatRng,
    Rng,
    TryRng,
};

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

impl TryRng for RRng {
    type Error = Infallible;
    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok((self.next_f64() * ((1_u64 << 32) as f64)) as u32)
    }
    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let high = self.next_u32() as u64;
        let low = self.next_u32() as u64;
        Ok((high << 32) | low)
    }
    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let mut chunks = dst.chunks_exact_mut(4);

        for chunk in &mut chunks {
            let arr: &mut [u8; 4] = chunk.try_into().expect("chunk to be exactly len 4");
            *arr = self.next_u32().to_le_bytes();
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let bytes = self.next_u32().to_le_bytes();
            for (i, r) in rem.iter_mut().enumerate() {
                *r = bytes[i];
            }
        }
        Ok(())
    }
}

impl FloatRng for RRng {
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // SAFETY: C-R interface
        unsafe { unif_rand() }
    }
}
