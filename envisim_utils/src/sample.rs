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

//! Sample conatiner

/// Sample container
#[must_use]
#[derive(Debug, Clone)]
pub struct Sample<ID> {
    /// The internal storage
    data: Vec<ID>,
    /// A flag for if the internal storage is in a sorted state
    sorted: bool,
}
impl<ID> Sample<ID> {
    /// Constructs a new sample container
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Sample {
            data: Vec::<ID>::with_capacity(capacity),
            sorted: true,
        }
    }
    /// Clears the container
    #[inline]
    pub fn clear(&mut self) { self.data.clear(); }
    /// Adds an index to the container
    #[inline]
    pub fn add(&mut self, id: ID) {
        self.data.push(id);
        self.sorted = false;
    }
    /// Sorts the indices in the continer
    #[inline]
    pub fn sort(&mut self)
    where
        ID: Ord,
    {
        self.data.sort_unstable();
        self.sorted = true;
    }
    /// Returns the sample indices as a slice
    #[must_use]
    #[inline]
    pub fn slice(&self) -> &[ID] { &self.data }
    /// Gets the length of the current sample
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }
    /// Returns `true` if the sample is empty
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    /// Returns `true` if the sample includes a unit
    #[expect(clippy::needless_pass_by_value, reason = "id assumed to be copy")]
    #[must_use]
    #[inline]
    pub fn contains(&self, id: ID) -> bool
    where
        ID: Ord,
    {
        if self.sorted {
            self.data.binary_search(&id).is_ok()
        } else {
            self.data.contains(&id)
        }
    }
    /// Sorts the container, and returns a clone of the internal container as a vector
    #[must_use]
    #[inline]
    pub fn to_sorted_vec(mut self) -> Vec<ID>
    where
        ID: Ord,
    {
        if !self.sorted {
            self.sort();
        }
        self.into()
    }
}

impl<ID> From<Sample<ID>> for Vec<ID> {
    #[inline]
    fn from(value: Sample<ID>) -> Self { value.data }
}
impl<ID> From<&Sample<ID>> for Vec<ID>
where
    ID: Clone,
{
    #[inline]
    fn from(value: &Sample<ID>) -> Self { value.data.clone() }
}
