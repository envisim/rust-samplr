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

//! List of indices

use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

use crate::random::RandomNumberGenerator;

/// A struct (list) for keeping track of indices in use. The internal list keeps track, without
/// order, of the indices.
#[derive(Clone, Debug)]
pub struct Indices {
    list: Vec<usize>,
    indices: FxHashMap<usize, usize>,
}

impl Indices {
    /// Constructs a new, empty `Indices`
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
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
    /// use envisim_utils::indices::Indices;
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
    /// use envisim_utils::indices::Indices;
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
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.list(), vec![0, 1, 2, 3]);
    /// ```
    #[inline]
    pub fn list(&self) -> &[usize] { &self.list }

    /// Returns a copy of the slice containing the indices
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// let v: Vec<usize> = il.to_vec();
    /// assert_eq!(il.list(), &v);
    /// ```
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> { self.list.to_vec() }

    /// Returns the index at position `k`, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.get(3).unwrap(), 3);
    /// assert_eq!(il.get(10), None);
    /// ```
    #[inline]
    pub fn get(&self, k: usize) -> Option<usize> { self.list.get(k).copied() }

    /// Returns the index at the first position, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.first().unwrap(), 0);
    /// ```
    #[inline]
    pub fn first(&self) -> Option<usize> { self.list.first().copied() }

    /// Returns the index at the last position, if any
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.last().unwrap(), 3);
    /// ```
    #[inline]
    pub fn last(&self) -> Option<usize> { self.list.last().copied() }

    /// Draws a random index from the list
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    /// use envisim_utils::random::*;
    ///
    /// let il = Indices::with_fill(4);
    /// let mut rng = SmallRng::seed_from_u64(4242);
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// ```
    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> Option<usize>
    where
        R: RandomNumberGenerator,
    {
        rng.relement(&self.list).copied()
    }

    /// Returns the next sequential unit after `from`, not including itself, if it exists.
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    /// use envisim_utils::random::*;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert_eq!(il.seq_after(0, 4), Some(1));
    /// assert_eq!(il.seq_after(3, 4), None);
    /// il.remove(2);
    /// assert_eq!(il.seq_after(1, 4), Some(3));
    /// ```
    #[inline]
    pub fn seq_after(&self, from: usize, max: usize) -> Option<usize> {
        ((from + 1)..max).find(|&id| self.contains(id))
    }

    /// Checks if the list contains an index
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert!(il.contains(3));
    /// assert!(!il.contains(4));
    /// ```
    #[inline]
    pub fn contains(&self, id: usize) -> bool { self.indices.contains_key(&id) }

    /// Inserts an index. Returns `None` if the index already exists
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.contains(42));
    /// assert!(il.insert(42).is_some());
    /// assert!(il.contains(42));
    /// assert!(il.insert(42).is_none());
    /// assert!(il.contains(42));
    /// ```
    #[inline]
    pub fn insert(&mut self, id: usize) -> Option<usize> {
        if self.contains(id) {
            return None;
        }

        self.list.push(id);
        let k = self.list.len() - 1;
        self.indices.insert(id, k);
        Some(k)
    }

    /// Returns the number of indices
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.len(), 4);
    /// ```
    #[inline]
    pub fn len(&self) -> usize { self.list.len() }
    /// Returns `true` if the list is empty
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.is_empty());
    /// il.clear();
    /// assert!(il.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool { self.list.is_empty() }

    /// Removes an index. Returns `None` if the index does not exist
    ///
    /// # Examples
    /// ```
    /// use envisim_utils::indices::Indices;
    ///
    /// let mut il = Indices::with_fill(4);
    /// assert!(il.contains(2));
    /// assert!(il.remove(2).is_some());
    /// assert!(il.remove(2).is_none());
    /// assert!(!il.contains(2));
    #[inline]
    pub fn remove(&mut self, id: usize) -> Option<usize> {
        let k = self.indices.remove(&id)?;
        self.list.swap_remove(k);
        if k != self.list.len() {
            *self.indices.get_mut(&self.list[k]).unwrap() = k;
        }
        Some(id)
    }
}

#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum Pair {
    #[default]
    Zero,
    One(usize),
    Two(usize, usize),
    More(usize, usize),
}
impl Pair {
    pub fn new((id1, id2): (usize, usize)) -> Self { Self::More(id1, id2) }
    pub fn is_zero(&self) -> bool { matches!(self, Self::Zero) }
    pub fn is_one(&self) -> bool { matches!(self, Self::One(..)) }
    pub fn is_two(&self) -> bool { matches!(self, Self::Two(..)) }
    pub fn is_more(&self) -> bool { matches!(self, Self::More(..)) }
    pub fn is_full(&self) -> bool { matches!(self, Self::Two(..) | Self::More(..)) }
}
impl From<&Indices> for Pair {
    fn from(indices: &Indices) -> Self {
        use Pair::*;
        let len = indices.len();
        if len == 0 {
            return Zero;
        }
        let id1 = indices.list()[0];
        if len == 1 {
            return One(id1);
        }
        let id2 = indices.list()[1];
        if len == 2 {
            return Two(id1, id2);
        }
        More(id1, id2)
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
