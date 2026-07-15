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

//! Balancing options container

/// Balancing options
#[must_use]
#[derive(Clone, Debug)]
pub struct BalancingOptions<P> {
    /// Balancing data
    data: P,
}

impl<P> From<P> for BalancingOptions<P> {
    #[inline]
    fn from(value: P) -> Self { Self::new(value) }
}

impl<P> BalancingOptions<P> {
    /// Returns a reference to the stored data
    #[inline]
    pub fn data(&self) -> &P { &self.data }
    /// Constructs a new options container for some data
    #[inline]
    pub fn new(data: P) -> Self { Self { data } }
}
