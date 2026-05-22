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

/// Sample container
#[must_use]
#[derive(Debug, Clone)]
pub struct Sample {
    /// The internal storage
    data: Vec<usize>,
    /// A flag for if the internal storage is in a sorted state
    sorted: bool,
}
impl Sample {
    /// Constructs a new sample container
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Sample {
            data: Vec::<usize>::with_capacity(capacity),
            sorted: true,
        }
    }
    /// Clears the container
    #[inline]
    pub fn clear(&mut self) { self.data.clear(); }
    /// Adds an index to the container
    #[inline]
    pub fn add(&mut self, idx: usize) {
        self.data.push(idx);
        self.sorted = false;
    }
    /// Sorts the indices in the continer
    #[inline]
    pub fn sort(&mut self) {
        self.data.sort_unstable();
        self.sorted = true;
    }
    /// Returns the sample indices as a slice
    #[must_use]
    #[inline]
    pub fn get(&self) -> &[usize] { &self.data }
    /// Gets the length of the current sample
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize { self.data.len() }
    /// Returns `true` if the sample is empty
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    /// Returns `true` if the sample includes a unit
    #[must_use]
    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        if self.sorted {
            self.data.binary_search(&idx).is_ok()
        } else {
            self.data.contains(&idx)
        }
    }
    /// Sorts the container, and returns a clone of the internal container as a vector
    #[must_use]
    #[inline]
    pub fn to_sorted_vec(mut self) -> Vec<usize> {
        if !self.sorted {
            self.sort();
        }
        self.into()
    }
}
impl From<Sample> for Vec<usize> {
    #[inline]
    fn from(value: Sample) -> Self { value.data }
}
impl From<&Sample> for Vec<usize> {
    #[inline]
    fn from(value: &Sample) -> Self { value.data.clone() }
}
