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

//! Views for sliceables
//!
//! Mirroring `AsRef` and `AsMut`, but using associated type.

use std::borrow::Cow;
use std::collections::{
    HashMap,
    HashSet,
};
use std::fmt::Debug;
use std::hash::{
    BuildHasher,
    Hash,
};
use std::rc::Rc;
use std::sync::Arc;

/// General data container
pub trait DataView {
    /// The type of the IDs
    type Id: Sized + Eq + Hash + Ord + Copy + Debug;
    /// The type of the stored values
    type Value;
    /// Iterator over data ids
    #[must_use]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone;
    /// Returns an iterator to the internal data.
    #[must_use]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone;
    /// Returns an iterator over the key-value pairs
    #[must_use]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone;
    /// Returns `true` if `id` exists in the container
    #[must_use]
    fn contains(&self, id: Self::Id) -> bool;
    /// Returns a reference to the value at `id`, if it exist.
    #[must_use]
    fn get(&self, id: Self::Id) -> Option<&Self::Value>;
    /// Returns the number of stored elements in the container.
    #[must_use]
    fn len(&self) -> usize;
    /// Returns `true` if the container is empty
    #[inline]
    #[must_use]
    fn is_empty(&self) -> bool { self.len() == 0 }
}
/// General mutable data container
pub trait DataViewMut: DataView {
    /// Returns a mutable iterator the internal data.
    #[must_use]
    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Self::Value>;
    /// Returns a mutable iterator the key-value pairs.
    #[must_use]
    fn entries_mut(&mut self) -> impl ExactSizeIterator<Item = (Self::Id, &mut Self::Value)>;
    /// Returns a mutable reference to the values stored at `id`, if it exists.
    #[must_use]
    fn get_mut(&mut self, id: Self::Id) -> Option<&mut Self::Value>;
}
/// Data container with contiguous layout
pub trait ContiguousDataView: DataView<Id = usize> {}
/// Data container that can be sliced
pub trait SliceView: ContiguousDataView {
    /// Returns a reference (view) to the internal data.
    #[must_use]
    fn slice(&self) -> &[Self::Value];
}
/// Mutable data container that can be sliced
pub trait SliceViewMut: SliceView + DataViewMut {
    /// Returns a mutable reference to the internal data.
    #[must_use]
    fn slice_mut(&mut self) -> &mut [Self::Value];
}
/// Re-constructable data container
pub trait ConstructableDataView: DataView {
    /// A similar type that can be used to re-construct the data container for different values
    type ConstructableContainer<V>: DataViewMut<Id = Self::Id, Value = V>;
    /// Construct a new container from an iterator
    #[must_use]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>;
    /// Construct a new container from a fallible iterator
    /// # Errors
    /// Returns an error if any item in the iterator returns an error.
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>;
    /// Map values of self to a new container
    #[must_use]
    #[inline]
    fn iter_map<V, F>(&self, f: F) -> Self::ConstructableContainer<V>
    where
        F: FnMut((Self::Id, &Self::Value)) -> (Self::Id, V),
    {
        Self::from_iter(self.entries().map(f))
    }
}

// REF IMPL
impl<T> DataView for &T
where
    T: DataView + ?Sized,
{
    type Id = T::Id;
    type Value = T::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { (**self).ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { (**self).values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        (**self).entries()
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { (**self).contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { (**self).get(id) }
    #[inline]
    fn len(&self) -> usize { (**self).len() }
    #[inline]
    fn is_empty(&self) -> bool { (**self).is_empty() }
}
impl<T> DataView for &mut T
where
    T: DataView + ?Sized,
{
    type Id = T::Id;
    type Value = T::Value;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { (**self).ids() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { (**self).values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        (**self).entries()
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { (**self).contains(id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { (**self).get(id) }
    #[inline]
    fn len(&self) -> usize { (**self).len() }
    #[inline]
    fn is_empty(&self) -> bool { (**self).is_empty() }
}
impl<T> DataViewMut for &mut T
where
    T: DataViewMut + ?Sized,
{
    #[inline]
    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Self::Value> {
        (**self).values_mut()
    }
    #[inline]
    fn entries_mut(&mut self) -> impl ExactSizeIterator<Item = (Self::Id, &mut Self::Value)> {
        (**self).entries_mut()
    }
    #[inline]
    fn get_mut(&mut self, id: Self::Id) -> Option<&mut Self::Value> { (**self).get_mut(id) }
}
impl<T> ContiguousDataView for &T where T: ContiguousDataView + ?Sized {}
impl<T> ContiguousDataView for &mut T where T: ContiguousDataView + ?Sized {}
impl<T> SliceView for &T
where
    T: SliceView + ?Sized,
{
    #[inline]
    fn slice(&self) -> &[Self::Value] { (**self).slice() }
}
impl<T> SliceView for &mut T
where
    T: SliceView + ?Sized,
{
    #[inline]
    fn slice(&self) -> &[Self::Value] { (**self).slice() }
}
impl<T> SliceViewMut for &mut T
where
    T: SliceViewMut + ?Sized,
{
    #[inline]
    fn slice_mut(&mut self) -> &mut [Self::Value] { (**self).slice_mut() }
}
impl<T> ConstructableDataView for &T
where
    T: ConstructableDataView + ?Sized,
{
    type ConstructableContainer<V> = T::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        T::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        T::try_from_iter(iter)
    }
}
impl<T> ConstructableDataView for &mut T
where
    T: ConstructableDataView + ?Sized,
{
    type ConstructableContainer<V> = T::ConstructableContainer<V>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        T::from_iter(iter)
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        T::try_from_iter(iter)
    }
}

/// Implements `SliceView`
macro_rules! impl_sliceview_self {
    (($($gens:tt)*), $ty:ty) => {
        impl<$($gens)*> SliceView for $ty {
            #[inline]
            fn slice(&self) -> &[Self::Value] { self.as_ref() }
        }
        impl_dataview_for_sliceview!(($($gens)*), $ty);
    };
    (($($gens:tt)*), $ty:ty, $cty:ty) => {
        impl_sliceview_self!(($($gens)*), $ty);
        impl<$($gens)*> ConstructableDataView for $ty {
            type ConstructableContainer<V> = $cty;
            #[inline]
            fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
            where
                I: Iterator<Item = (Self::Id, V)>,
            {
                iter.map(|(_, v)| v).collect()
            }
            #[inline]
            fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
            where
                I: Iterator<Item = Result<(Self::Id, V), E>>,
            {
                iter.map(|r| r.map(|(_, v)| v)).collect()
            }
        }
    };
}
/// Implements `SliceViewMut`
macro_rules! impl_sliceviewmut_self {
    (($($gens:tt)*), $ty:ty) => {
        impl<$($gens)*> SliceViewMut for $ty {
            #[inline]
            fn slice_mut(&mut self) -> &mut [Self::Value] { self.as_mut() }
        }
        impl_dataviewmut_for_sliceviewmut!(($($gens)*), $ty);
    }
}
/// Implements `DataView` for `SliceView`-ables
macro_rules! impl_dataview_for_sliceview {
    (($($gens:tt)*), $ty:ty) => {
        impl<$($gens)*> ContiguousDataView for $ty {}
        impl<$($gens)*> DataView for $ty {
            type Id = usize;
            type Value = N;
            #[inline]
            fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { 0..self.len() }
            #[inline]
            fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone {
                self.slice().iter()
            }
            #[inline]
            fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
                self.slice().iter().enumerate()
            }
            #[inline]
            fn contains(&self, id: Self::Id) -> bool { id < self.len() }
            #[inline]
            fn get(&self, id: Self::Id) -> Option<&Self::Value> { self.slice().get(id) }
            #[inline]
            fn len(&self) -> usize { self.slice().len() }
            #[inline]
            fn is_empty(&self) -> bool {self.slice().is_empty()}
        }
    };
}
/// Implements `DataViewMut` for `SliceViewMut`-ables
macro_rules! impl_dataviewmut_for_sliceviewmut {
    (($($gens:tt)*), $ty:ty) => {
        impl<$($gens)*> DataViewMut for $ty {
            #[inline]
            fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Self::Value> {
                self.slice_mut().iter_mut()
            }
            #[inline]
            fn entries_mut(&mut self) -> impl ExactSizeIterator<Item = (Self::Id, &mut Self::Value)> {
                self.slice_mut().iter_mut().enumerate()
            }
            #[inline]
            fn get_mut(&mut self, id: Self::Id) -> Option<&mut Self::Value> {
                self.slice_mut().get_mut(id)
            }
        }
    };
}

impl_sliceview_self!((N, const L: usize), [N; L], Box<[V]>);
impl_sliceviewmut_self!((N, const L: usize), [N; L]);

impl_sliceview_self!((N), [N], Box<[V]>);
impl_sliceviewmut_self!((N), [N]);

impl_sliceview_self!((N), Vec<N>, Vec<V>);
impl_sliceviewmut_self!((N), Vec<N>);

impl_sliceview_self!((N), Box<[N]>, Box<[V]>);
impl_sliceviewmut_self!((N), Box<[N]>);

impl_sliceview_self!((N: Clone), Cow<'_, [N]>, Box<[V]>);
impl<N: Clone> SliceViewMut for Cow<'_, [N]> {
    #[inline]
    fn slice_mut(&mut self) -> &mut [Self::Value] { self.to_mut() }
}
impl_dataviewmut_for_sliceviewmut!((N: Clone), Cow<'_, [N]>);

impl_sliceview_self!((N), Rc<[N]>, Box<[V]>);

impl_sliceview_self!((N), Arc<[N]>, Box<[V]>);

impl<K, V, S> DataView for HashMap<K, V, S>
where
    K: Sized + Eq + Hash + Ord + Copy + Debug,
    S: BuildHasher,
{
    type Id = K;
    type Value = V;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.keys().copied() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { self.values() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.iter().map(|(k, v)| (*k, v))
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { self.contains_key(&id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { self.get(&id) }
    #[inline]
    fn len(&self) -> usize { self.len() }
    #[inline]
    fn is_empty(&self) -> bool { self.is_empty() }
}
impl<K, V, S> DataViewMut for HashMap<K, V, S>
where
    K: Sized + Eq + Hash + Ord + Copy + Debug,
    S: BuildHasher,
{
    #[inline]
    fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Self::Value> {
        self.values_mut()
    }
    #[inline]
    fn entries_mut(&mut self) -> impl ExactSizeIterator<Item = (Self::Id, &mut Self::Value)> {
        self.iter_mut().map(|(k, v)| (*k, v))
    }
    #[inline]
    fn get_mut(&mut self, id: Self::Id) -> Option<&mut Self::Value> { self.get_mut(&id) }
}
impl<K, V0, S> ConstructableDataView for HashMap<K, V0, S>
where
    K: Sized + Eq + Hash + Ord + Copy + Debug,
    S: BuildHasher + Default,
{
    type ConstructableContainer<V> = HashMap<K, V, S>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        iter.collect()
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        iter.collect()
    }
}

impl<K, S> DataView for HashSet<K, S>
where
    K: Sized + Eq + Hash + Ord + Copy + Debug,
    S: BuildHasher,
{
    type Id = K;
    type Value = K;
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { self.iter().copied() }
    #[inline]
    fn values(&self) -> impl ExactSizeIterator<Item = &Self::Value> + Clone { self.iter() }
    #[inline]
    fn entries(&self) -> impl ExactSizeIterator<Item = (Self::Id, &Self::Value)> + Clone {
        self.iter().map(|k| (*k, k))
    }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { HashSet::contains(self, &id) }
    #[inline]
    fn get(&self, id: Self::Id) -> Option<&Self::Value> { HashSet::get(self, &id) }
    #[inline]
    fn len(&self) -> usize { self.len() }
    #[inline]
    fn is_empty(&self) -> bool { self.is_empty() }
}
impl<K, S> ConstructableDataView for HashSet<K, S>
where
    K: Sized + Eq + Hash + Ord + Copy + Debug,
    S: BuildHasher + Default,
{
    type ConstructableContainer<V> = HashMap<K, V, S>;
    #[inline]
    fn from_iter<I, V>(iter: I) -> Self::ConstructableContainer<V>
    where
        I: Iterator<Item = (Self::Id, V)>,
    {
        iter.collect()
    }
    #[inline]
    fn try_from_iter<I, V, E>(iter: I) -> Result<Self::ConstructableContainer<V>, E>
    where
        I: Iterator<Item = Result<(Self::Id, V), E>>,
    {
        iter.collect()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::{
        HashMap,
        HashSet,
    };
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_slice_view_vec_and_slice() {
        let mut vec = vec![10, 20, 30];

        // DataView and SliceView
        assert_eq!(vec.len(), 3);
        assert!(!vec.is_empty());
        assert_eq!(vec.get(1), Some(&20));
        assert_eq!(vec.get(5), None);
        assert!(vec.contains(0));
        assert!(!vec.contains(3));
        assert_eq!(vec.slice(), &[10, 20, 30]);

        // DataViewMut & SliceViewMut
        if let Some(v) = vec.get_mut(1) {
            *v = 25;
        }
        assert_eq!(vec.slice(), &[10, 25, 30]);

        vec.slice_mut()[2] = 35;
        assert_eq!(vec.slice(), &[10, 25, 35]);

        // Iterators
        let ids: Vec<_> = vec.ids().collect();
        assert_eq!(ids, vec![0, 1, 2]);

        let vals: Vec<_> = vec.values().copied().collect();
        assert_eq!(vals, vec![10, 25, 35]);

        let entries: Vec<_> = vec.entries().collect();
        assert_eq!(entries, vec![(0, &10), (1, &25), (2, &35)]);

        vec.values_mut().for_each(|v| *v += 2);
        assert_eq!(vec, vec![12, 27, 37]);
    }

    #[test]
    fn test_array_and_box() {
        let arr = [1, 2, 3];
        assert_eq!(arr.slice(), &[1, 2, 3]);
        assert_eq!(arr.get(1), Some(&2));

        let mut boxed: Box<[i32]> = Box::new([4, 5, 6]);
        assert_eq!(boxed.slice(), &[4, 5, 6]);
        boxed.slice_mut()[0] = 40;
        assert_eq!(boxed.get(0), Some(&40));
    }

    #[test]
    fn test_reference_wrappers() {
        let mut vec = vec![1, 2, 3];

        let shared_ref = &vec;
        assert_eq!(shared_ref.get(0), Some(&1));
        assert_eq!(shared_ref.slice(), &[1, 2, 3]);

        let mut_ref = &mut vec;
        if let Some(v) = mut_ref.get_mut(0) {
            *v = 10;
        }
        assert_eq!(mut_ref.slice(), &[10, 2, 3]);
    }

    #[test]
    fn test_cow_rc_arc() {
        let cow: Cow<'_, [i32]> = Cow::Borrowed(&[1, 2, 3]);
        assert_eq!(cow.slice(), &[1, 2, 3]);

        let mut cow_mut = cow.clone();
        cow_mut.slice_mut()[0] = 99;
        assert_eq!(cow_mut.slice(), &[99, 2, 3]);

        let rc: Rc<[i32]> = Rc::from(vec![10, 20]);
        assert_eq!(rc.slice(), &[10, 20]);

        let arc: Arc<[i32]> = Arc::from(vec![30, 40]);
        assert_eq!(arc.slice(), &[30, 40]);
    }

    #[test]
    fn test_hashmap_view() {
        let mut map = HashMap::new();
        map.insert("a", 10);
        map.insert("b", 20);

        assert_eq!(map.len(), 2);
        assert!(map.contains("a"));
        assert_eq!(map.get("a"), Some(&10));

        if let Some(v) = map.get_mut("a") {
            *v = 15;
        }
        assert_eq!(map.get("a"), Some(&15));

        // ConstructableDataView / iter_map
        let mapped = map.iter_map(|(k, v)| (k, v * 2));
        assert_eq!(mapped.get("a"), Some(&30));
        assert_eq!(mapped.get("b"), Some(&40));
    }

    #[test]
    fn test_hashset_view() {
        let mut set = HashSet::new();
        set.insert(1);
        set.insert(2);

        assert_eq!(set.len(), 2);
        assert!(DataView::contains(&set, 1));
        assert_eq!(DataView::get(&set, 1), Some(&1));
    }

    #[test]
    fn test_constructable_from_and_try_from_iter() {
        // Vec / Slice constructable
        let data: HashMap<i32, i32> = vec![(10, 100), (20, 200)].into_iter().collect();
        let constructed = data.iter_map(|(id, v)| (id + 2, *v * 2));
        let mut result: Vec<_> = constructed.entries().map(|(id, v)| (id, *v)).collect();
        result.sort_by_key(|v| v.0);
        assert_eq!(result.slice(), &[(12, 200), (22, 400)]);

        // Fallible iteration
        let ok_items = vec![Ok((0, 10)), Ok((1, 20))];
        let res: Result<Vec<i32>, &str> = Vec::<i32>::try_from_iter(ok_items.into_iter());
        assert_eq!(res, Ok(vec![10, 20]));

        let err_items = vec![Ok((0, 10)), Err("failure")];
        let err_res: Result<Vec<i32>, &str> = Vec::<i32>::try_from_iter(err_items.into_iter());
        assert_eq!(err_res, Err("failure"));
    }
}
