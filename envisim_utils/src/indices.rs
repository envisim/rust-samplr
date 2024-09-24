// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

//! List of indices

use crate::utils::random_element;
use rand::Rng;
use rustc_hash::{FxBuildHasher, FxHashMap};

/// A struct (list) for keeping track of indices in use. The internal list keeps track, without
/// order, of the indices.
pub struct Indices {
    list: Vec<usize>,
    indices: FxHashMap<usize, usize>,
}

impl Indices {
    /// Constructs a new, empty `Indices`
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    /// let il = Indices::new(10);
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Indices {
            list: Vec::<usize>::with_capacity(capacity),
            indices: FxHashMap::<usize, usize>::with_capacity_and_hasher(capacity, FxBuildHasher),
        }
    }

    /// Constructs a new `Indices`, filled with (0..length)
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    /// let il = Indices::with_fill(10);
    /// ```
    #[inline]
    pub fn with_fill(length: usize) -> Self {
        Indices {
            list: (0..length).collect::<Vec<usize>>(),
            indices: (0..length)
                .map(|v| (v, v))
                .collect::<FxHashMap<usize, usize>>(),
        }
    }

    /// Clears the list of indices
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let mut il = Indices::with_fill(10);
    /// il.clear();
    /// assert_eq!(il.len(), 0);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.list.clear();
        self.indices.clear();
    }

    /// Returns a reference to the slice containing the indicies
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.list(), vec![0, 1, 2, 3]);
    /// ```
    #[inline]
    pub fn list(&self) -> &[usize] {
        &self.list
    }

    /// Returns a copy of the slice containing the indices
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// let v: Vec<usize> = il.to_vec();
    /// assert_eq!(il.list(), &v);
    /// ```
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.list.to_vec()
    }

    /// Returns the index at position `k`, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.get(3).unwrap(), &3);
    /// assert_eq!(il.get(10), None);
    /// ```
    #[inline]
    pub fn get(&self, k: usize) -> Option<&usize> {
        self.list.get(k)
    }

    /// Returns the index at the first position, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.first().unwrap(), &0);
    /// ```
    #[inline]
    pub fn first(&self) -> Option<&usize> {
        self.list.first()
    }

    /// Returns the index at the last position, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.last().unwrap(), &3);
    /// ```
    #[inline]
    pub fn last(&self) -> Option<&usize> {
        self.list.last()
    }

    /// Draws a random index from the list
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    /// use rand::{rngs::SmallRng, SeedableRng};
    ///
    /// let il = Indices::with_fill(4);
    /// let mut rng = SmallRng::seed_from_u64(4242);
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// ```
    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> Option<&usize>
    where
        R: Rng + ?Sized,
    {
        random_element(rng, &self.list)
    }

    /// Checks if the list contains an index
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert!(il.contains(3));
    /// assert!(!il.contains(4));
    /// ```
    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        self.indices.contains_key(&id)
    }

    /// Inserts an index. Returns [`IndicesError::GhostIndex`] if the index already exists
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.contains(42));
    /// assert!(il.insert(42).is_ok());
    /// assert!(il.contains(42));
    /// assert!(il.insert(42).is_err());
    /// assert!(il.contains(42));
    /// ```
    #[inline]
    pub fn insert(&mut self, id: usize) -> Result<usize, IndicesError> {
        if self.contains(id) {
            return Err(IndicesError::GhostIndex(id));
        }

        self.list.push(id);
        let k = self.list.len() - 1;
        self.indices.insert(id, k);
        Ok(k)
    }

    /// Returns the number of indices
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.len(), 4);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.list.len()
    }
    /// Returns `true` if the list is empty
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.is_empty());
    /// il.clear();
    /// assert!(il.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Removes an index. Returns [`IndicesError::GhostIndex`] if the index already exists
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(il.contains(2));
    /// assert!(il.remove(2).is_ok());
    /// assert!(il.remove(2).is_err());
    /// assert!(!il.contains(2));
    #[inline]
    pub fn remove(&mut self, id: usize) -> Result<(), IndicesError> {
        match self.indices.remove(&id) {
            Some(k) => {
                self.list.swap_remove(k);

                if k != self.list.len() {
                    *self.indices.get_mut(&self.list[k]).unwrap() = k;
                }

                Ok(())
            }
            None => Err(IndicesError::GhostIndex(id)),
        }
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum IndicesError {
    // Accessing non-existing index
    GhostIndex(usize),
    // Accessing out-of-bounds internal index
    OutOfBoundsK(usize),
}

impl std::error::Error for IndicesError {}

impl std::fmt::Display for IndicesError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            IndicesError::GhostIndex(id) => {
                write!(f, "cannot access non-existing index {id}")
            }
            IndicesError::OutOfBoundsK(k) => {
                write!(f, "cannot access out-of-bounds k-index {k}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let il = Indices::new(10);
        assert!(il.list.capacity() >= 10);
        assert!(il.indices.capacity() >= 10);
    }

    #[test]
    fn with_fill() {
        let il = Indices::with_fill(4);
        assert_eq!(il.list(), vec![0, 1, 2, 3]);
    }
}
