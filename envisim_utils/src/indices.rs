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

//! List of indices

use std::num::NonZeroUsize;
use std::ops::Index;

use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

use crate::random::{
    Rng,
    random_element,
};

/// A struct (list) for keeping track of indices in use. The internal list keeps track, without
/// order, of the indices.
#[must_use]
#[derive(Clone, Debug)]
pub struct Indices {
    /// The remaining indices mapping to their position in the list
    indices: FxHashMap<usize, usize>,
    /// An (unordered) set of remaining indices
    list: Vec<usize>,
}

impl Indices {
    /// Constructs a new, empty `Indices`
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::new(10);
    /// assert!(il.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Indices {
            list: Vec::<usize>::with_capacity(capacity),
            indices: FxHashMap::<usize, usize>::with_capacity_and_hasher(capacity, FxBuildHasher),
        }
    }

    /// Constructs a new `Indices`, filled with units `(0..length)`
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(10);
    /// assert_eq!(il.first(), Some(9));
    /// assert_eq!(il.last(), Some(0));
    /// ```
    #[inline]
    pub fn with_fill<NZ>(length: NZ) -> Self
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let Ok(nz) = length.try_into() else {
            return Self::new(0);
        };
        // By storing the list in reverse order, algos that need to be able to draw in order can
        // always draw from the end, and a swap remove will guarantee that the selected units are
        // stable, without incurring a higher cost needed in order to keep order in start of list.
        Indices {
            list: (0..nz.get()).rev().collect::<Vec<usize>>(),
            indices: (0..nz.get())
                .rev()
                .map(|v| (v, v))
                .collect::<FxHashMap<usize, usize>>(),
        }
    }

    /// Clears the list of indices
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(10);
    /// assert_eq!(il.len(), 10);
    /// il.clear();
    /// assert!(il.is_empty());
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
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.list(), vec![3, 2, 1, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn list(&self) -> &[usize] { &self.list }

    /// Returns a copy of the slice containing the indices
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(4);
    /// let v: Vec<usize> = il.to_vec();
    /// assert_eq!(v, vec![3, 2, 1, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn to_vec(&self) -> Vec<usize> { self.list.clone() }

    /// Returns the index at position `k`, if any
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.get(3), Some(0));
    /// assert_eq!(il.get(10), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&self, k: usize) -> Option<usize> { self.list.get(k).copied() }

    /// Returns the index at the first position, if any
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(4);
    /// assert_eq!(il.first(), Some(3));
    /// il.clear();
    /// assert_eq!(il.first(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn first(&self) -> Option<usize> { self.list.first().copied() }

    /// Returns the index at the last position, if any
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(4);
    /// assert_eq!(il.last(), Some(0));
    /// il.clear();
    /// assert_eq!(il.last(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn last(&self) -> Option<usize> { self.list.last().copied() }

    /// Draws a random index from the list
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// use envisim_utils::random::*;
    /// let il = Indices::with_fill(4);
    /// let mut rng = SmallRng::seed_from_u64(4242);
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// ```
    #[must_use]
    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> Option<usize>
    where
        R: Rng,
    {
        random_element(rng, &self.list).copied()
    }

    /// Checks if the list contains an index
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(4);
    /// assert!(il.contains(3));
    /// assert!(!il.contains(4));
    /// ```
    #[must_use]
    #[inline]
    pub fn contains(&self, id: usize) -> bool { self.indices.contains_key(&id) }

    /// Inserts an index. Returns `false` if the index already exists
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.contains(42));
    /// assert!(il.insert(42));
    /// assert!(il.contains(42));
    /// assert!(!il.insert(42));
    /// assert!(il.contains(42));
    /// ```
    #[inline]
    pub fn insert(&mut self, id: usize) -> bool {
        if self.contains(id) {
            return false;
        }

        self.list.push(id);
        let k = self.list.len() - 1;
        self.indices.insert(id, k);
        true
    }

    /// Returns the number of indices
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::with_fill(4);
    /// assert_eq!(il.len(), 4);
    /// ```
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize { self.list.len() }
    /// Returns `true` if the list is empty
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(4);
    /// assert!(!il.is_empty());
    /// il.clear();
    /// assert!(il.is_empty());
    /// ```
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.list.is_empty() }

    /// Removes an index. Returns `false` if the index does not exist
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::with_fill(4);
    /// assert!(il.contains(2));
    /// assert!(il.remove(2));
    /// assert!(!il.remove(2));
    /// assert!(!il.contains(2));
    #[expect(clippy::missing_panics_doc, reason = "infallible")]
    #[inline]
    pub fn remove(&mut self, id: usize) -> bool {
        let Some(k) = self.indices.remove(&id) else {
            return false;
        };
        self.list.swap_remove(k);
        if k != self.list.len() {
            *self
                .indices
                .get_mut(&self.list[k])
                .expect("indices to include the swapped unit") = k;
        }
        true
    }
}

impl Index<usize> for Indices {
    type Output = usize;
    #[must_use]
    #[inline]
    fn index(&self, index: usize) -> &Self::Output { &self.list[index] }
}

#[expect(
    clippy::exhaustive_enums,
    reason = "additional pair variants is a breaking change"
)]
#[must_use]
#[derive(Clone, Debug, Default)]
pub enum Pair {
    #[default]
    Zero,
    One(usize),
    Two(usize, usize),
    More(usize, usize),
}
impl Pair {
    #[inline]
    pub fn new((id1, id2): (usize, usize)) -> Self { Self::More(id1, id2) }
    #[must_use]
    #[inline]
    pub fn is_zero(&self) -> bool { matches!(*self, Self::Zero) }
    #[must_use]
    #[inline]
    pub fn is_one(&self) -> bool { matches!(*self, Self::One(..)) }
    #[must_use]
    #[inline]
    pub fn is_two(&self) -> bool { matches!(*self, Self::Two(..)) }
    #[must_use]
    #[inline]
    pub fn is_more(&self) -> bool { matches!(*self, Self::More(..)) }
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool { matches!(*self, Self::Two(..) | Self::More(..)) }
}
impl From<&Indices> for Pair {
    #[inline]
    fn from(indices: &Indices) -> Self {
        match indices.list() {
            [b, a] => Pair::Two(*a, *b),
            // Matches at least two
            [.., b, a] => Pair::More(*a, *b),
            [a] => Pair::One(*a),
            [] => Pair::Zero,
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
        assert_eq!(il.list(), vec![3, 2, 1, 0]);
    }
}
