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

//! Coordination options container

use std::borrow::Cow;
use std::convert::AsRef;
use std::num::NonZeroUsize;

use super::error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::random::RandomNumberGenerator;

#[must_use]
#[derive(Clone, Debug)]
pub struct CoordinationOptions<'bcoord> {
    /// Random value data
    data: Option<Cow<'bcoord, [f64]>>,
}

impl<'bcoord> CoordinationOptions<'bcoord> {
    /// Returns the value at `id`, or `None` if no values exist, or the `id` does not exist.
    #[must_use]
    #[inline]
    pub fn get(&self, id: usize) -> Option<f64> {
        self.data.as_ref().and_then(|dt| dt.get(id).copied())
    }
    #[must_use]
    #[inline]
    pub fn get_or<R>(&self, id: usize, rng: &mut R) -> f64
    where
        R: RandomNumberGenerator,
    {
        self.get(id).unwrap_or_else(|| rng.rf64())
    }
    /// Returns the stored random values
    #[must_use]
    #[inline]
    pub fn data(&'bcoord self) -> Option<&'bcoord [f64]> { self.data.as_ref().map(AsRef::as_ref) }
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_none() }
    /// # Errors
    /// If the provided `len` is less than the number of values stored, returns an error.
    /// Usable for ensuring that enough random values are provided.
    #[inline]
    pub fn check(&self, len: NonZeroUsize) -> SamplingOptionsResult<()> {
        if let Some(dt) = self.data.as_ref() {
            if dt.len() < len.get() {
                return Err(SamplingOptionsError::InvalidRandomValues);
            }
        }
        Ok(())
    }
    /// Constructs a new random value container
    #[inline]
    pub fn new(data: Cow<'bcoord, [f64]>) -> Self { Self { data: Some(data) } }
    /// Constructs a new random value container
    #[inline]
    pub fn new_empty() -> Self { Self { data: None } }
}
impl<'bcoord> From<Cow<'bcoord, [f64]>> for CoordinationOptions<'bcoord> {
    #[inline]
    fn from(value: Cow<'bcoord, [f64]>) -> Self { Self::new(value) }
}
impl<'bcoord> From<&'bcoord [f64]> for CoordinationOptions<'bcoord> {
    #[inline]
    fn from(value: &'bcoord [f64]) -> Self { Self::new(Cow::Borrowed(value)) }
}
impl From<Vec<f64>> for CoordinationOptions<'_> {
    #[inline]
    fn from(value: Vec<f64>) -> Self { Self::new(Cow::Owned(value)) }
}
