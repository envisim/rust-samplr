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

use std::num::NonZeroUsize;

use super::error::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::utils::SliceView;

#[must_use]
#[derive(Clone, Debug)]
pub struct CoordinationRandomValues<CD> {
    /// Random value data
    data: CD,
}

impl<CD> CoordinationRandomValues<CD> {
    /// Constructs a new random value container
    #[inline]
    pub fn new(data: CD) -> Self { Self { data } }
    /// Returns a reference to the internal data
    #[must_use]
    #[inline]
    pub fn data(&self) -> &[CD::Elem]
    where
        CD: SliceView,
    {
        self.data.data()
    }
    /// Returns the value at `id`, or `None` if no values exist, or the `id` does not exist.
    #[must_use]
    #[inline]
    pub fn get(&self, id: usize) -> Option<CD::Elem>
    where
        CD: SliceView,
        CD::Elem: Copy,
    {
        self.data.data().get(id).copied()
    }
    /// Checks if a `CoordinationOptions` is valid
    /// # Errors
    /// If the provided `len` is less than the number of values stored, returns an error.
    /// Usable for ensuring that enough random values are provided.
    #[inline]
    pub fn check(&self, len: NonZeroUsize) -> SamplingOptionsResult<()>
    where
        CD: SliceView,
    {
        if self.data.data().len() < len.get() {
            Err(SamplingOptionsError::InvalidRandomValues)
        } else {
            Ok(())
        }
    }
}

impl<CD> From<CD> for CoordinationRandomValues<CD>
where
    CD: SliceView,
{
    #[inline]
    fn from(data: CD) -> Self { Self::new(data) }
}
