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

//! List of indices.

use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::Index;

use rustc_hash::{
    FxBuildHasher,
    FxHashMap,
};

use crate::random::{
    Rand,
    random_element,
};

/// A struct (list) for keeping track of indices in use. The internal list keeps track, without
/// order, of the indices.
#[must_use]
#[derive(Clone, Debug)]
pub struct Indices<ID> {
    /// The remaining indices mapping to their position in the list.
    indices: FxHashMap<ID, usize>,
    /// An (unordered) set of remaining indices.
    list: Vec<ID>,
}

impl<ID> Indices<ID> {
    /// Constructs a new, empty `Indices`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::<usize>::new(10);
    /// assert!(il.is_empty());
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Indices {
            list: Vec::<ID>::with_capacity(capacity),
            indices: FxHashMap::<ID, usize>::with_capacity_and_hasher(capacity, FxBuildHasher),
        }
    }
    /// Constructs a new index list from an iterator of ids.
    ///
    /// Returns `None` if the ids contains duplicates.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::try_from_iter(vec![10, 8, 2]).unwrap();
    /// assert!(il.contains(8));
    /// ```
    #[inline]
    pub fn try_from_iter<I>(ids: I) -> Option<Self>
    where
        ID: Copy + Eq + Hash,
        I: IntoIterator<Item = ID, IntoIter: ExactSizeIterator>,
    {
        let ids = ids.into_iter();
        let mut il = Self::new(ids.len());
        for (k, id) in ids.enumerate() {
            il.list.push(id);
            if il.indices.insert(id, k).is_some() {
                return None;
            }
        }
        Some(il)
    }

    /// Clears the list of indices.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.len(), 4);
    /// il.clear();
    /// assert!(il.is_empty());
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.list.clear();
        self.indices.clear();
    }

    /// Returns a reference to the slice containing the indicies.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.list(), vec![3, 2, 1, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn list(&self) -> &[ID] { &self.list }

    /// Returns a copy of the slice containing the indices.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// let v: Vec<usize> = il.to_vec();
    /// assert_eq!(v, vec![3, 2, 1, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn to_vec(&self) -> Vec<ID>
    where
        ID: Clone,
    {
        self.list.clone()
    }

    /// Returns the index at position `k`, if any.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.get(3), Some(0));
    /// assert_eq!(il.get(10), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn get(&self, k: usize) -> Option<ID>
    where
        ID: Copy,
    {
        self.list.get(k).copied()
    }

    /// Returns the index at the first position, if any.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.first(), Some(3));
    /// il.clear();
    /// assert_eq!(il.first(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn first(&self) -> Option<ID>
    where
        ID: Copy,
    {
        self.list.first().copied()
    }

    /// Returns the index at the last position, if any.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.last(), Some(0));
    /// il.clear();
    /// assert_eq!(il.last(), None);
    /// ```
    #[must_use]
    #[inline]
    pub fn last(&self) -> Option<ID>
    where
        ID: Copy,
    {
        self.list.last().copied()
    }

    /// Draws a random index from the list.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// use envisim_utils::random::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// let mut rng = SmallRng::seed_from_u64(4242);
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// assert!(il.draw(&mut rng).is_some());
    /// ```
    #[must_use]
    #[inline]
    pub fn draw<R>(&self, rng: &mut R) -> Option<ID>
    where
        ID: Copy,
        R: Rand<usize>,
    {
        random_element(rng, &self.list).copied()
    }

    /// Checks if the list contains an index.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert!(il.contains(3));
    /// assert!(!il.contains(4));
    /// ```
    #[expect(clippy::needless_pass_by_value, reason = "id assumed to be copy")]
    #[must_use]
    #[inline]
    pub fn contains(&self, id: ID) -> bool
    where
        ID: Eq + Hash,
    {
        self.indices.contains_key(&id)
    }

    /// Inserts an index. Returns `false` if the index already exists.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert!(!il.contains(42));
    /// assert!(il.insert(42));
    /// assert!(il.contains(42));
    /// assert!(!il.insert(42));
    /// assert!(il.contains(42));
    /// ```
    #[inline]
    pub fn insert(&mut self, id: ID) -> bool
    where
        ID: Copy + Eq + Hash,
    {
        if self.contains(id) {
            return false;
        }

        self.list.push(id);
        let k = self.list.len() - 1;
        self.indices.insert(id, k);
        true
    }

    /// Returns the number of indices.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert_eq!(il.len(), 4);
    /// ```
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize { self.list.len() }
    /// Returns `true` if the list is empty.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert!(!il.is_empty());
    /// il.clear();
    /// assert!(il.is_empty());
    /// ```
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool { self.list.is_empty() }

    /// Removes an index. Returns `false` if the index does not exist.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let mut il = Indices::from_len(std::num::NonZeroUsize::new(4).unwrap());
    /// assert!(il.contains(2));
    /// assert!(il.remove(2));
    /// assert!(!il.remove(2));
    /// assert!(!il.contains(2));
    #[expect(clippy::missing_panics_doc, reason = "infallible")]
    #[expect(clippy::needless_pass_by_value, reason = "id assumed to be copy")]
    #[inline]
    pub fn remove(&mut self, id: ID) -> bool
    where
        ID: Eq + Hash,
    {
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

impl Indices<usize> {
    /// Constructs a new `Indices`, filled with units `(0..length)`.
    ///
    /// # Examples
    /// ```
    /// # use envisim_utils::indices::*;
    /// let il = Indices::from_len(std::num::NonZeroUsize::new(10).unwrap());
    /// assert_eq!(il.first(), Some(9));
    /// assert_eq!(il.last(), Some(0));
    /// ```
    #[inline]
    pub fn from_len(length: NonZeroUsize) -> Self {
        // By storing the list in reverse order, algos that need to be able to draw in order can
        // always draw from the end, and a swap remove will guarantee that the selected units are
        // stable, without incurring a higher cost needed in order to keep order in start of list.
        Indices {
            list: (0..length.get()).rev().collect::<Vec<usize>>(),
            indices: (0..length.get())
                .rev()
                .map(|v| (v, v))
                .collect::<FxHashMap<usize, usize>>(),
        }
    }
}

impl<ID> Index<usize> for Indices<ID> {
    type Output = ID;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output { &self.list[index] }
}

/// A pair of indices, with the enum indicating the number of indices remaining.
#[expect(
    clippy::exhaustive_enums,
    reason = "additional pair variants is a breaking change"
)]
#[must_use]
#[derive(Clone, Debug, Default)]
pub enum Pair<ID> {
    /// Zero indices remaining.
    #[default]
    Zero,
    /// Exactly one index remaining.
    One(ID),
    /// Exactly two indices remaining.
    Two(ID, ID),
    /// More than two indicies remaining.
    More(ID, ID),
}
impl<ID> Pair<ID> {
    /// Constructs a new `More` variant.
    #[inline]
    pub fn new((id1, id2): (ID, ID)) -> Self { Self::More(id1, id2) }
    /// Returns `true` if the `Pair` is `Zero`.
    #[must_use]
    #[inline]
    pub fn is_zero(&self) -> bool { matches!(*self, Self::Zero) }
    /// Returns `true` if the `Pair` is `One`.
    #[must_use]
    #[inline]
    pub fn is_one(&self) -> bool { matches!(*self, Self::One(..)) }
    /// Returns `true` if the `Pair` is `Two`.
    #[must_use]
    #[inline]
    pub fn is_two(&self) -> bool { matches!(*self, Self::Two(..)) }
    /// Returns `true` if the `Pair` is `More`.
    #[must_use]
    #[inline]
    pub fn is_more(&self) -> bool { matches!(*self, Self::More(..)) }
    /// Returns `true` if the `Pair` is `Two` or `More`.
    #[must_use]
    #[inline]
    pub fn is_full(&self) -> bool { matches!(*self, Self::Two(..) | Self::More(..)) }
}
impl<ID> From<&Indices<ID>> for Pair<ID>
where
    ID: Copy,
{
    #[inline]
    fn from(indices: &Indices<ID>) -> Self {
        match indices.list() {
            [b, a] => Pair::Two(*a, *b),
            // Matches at least two
            [.., b, a] => Pair::More(*a, *b),
            [a] => Pair::One(*a),
            [] => Pair::Zero,
        }
    }
}
impl<ID> From<(Option<ID>, Option<ID>)> for Pair<ID>
where
    ID: Copy,
{
    #[inline]
    fn from((id1, id2): (Option<ID>, Option<ID>)) -> Self {
        match (id1, id2) {
            (Some(id1), Some(id2)) => Pair::More(id1, id2),
            (Some(id1), None) => Pair::One(id1),
            (None, Some(id2)) => Pair::One(id2),
            _ => Pair::Zero,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let il = Indices::<usize>::new(10);
        assert!(il.list.capacity() >= 10);
        assert!(il.indices.capacity() >= 10);
    }

    #[test]
    fn with_fill() {
        let il = Indices::from_len(NonZeroUsize::new(4).unwrap());
        assert_eq!(il.list(), vec![3, 2, 1, 0]);
    }
}
