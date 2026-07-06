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

//! Provides the `PointSet` trait

use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroUsize;

use num_traits::ConstZero;

use super::Number;

/// Methods for accessing containers of points in some coordinate system.
/// The container must not be empty.
#[expect(clippy::len_without_is_empty, reason = "cannot be empty")]
pub trait PointSet {
    type Id: Sized + Eq + Hash + Ord + Copy + Debug;
    type Value: Number;

    /// Number of points in set
    #[must_use]
    fn len(&self) -> NonZeroUsize;
    /// Iterator over point ids
    #[must_use]
    fn ids(&self) -> impl Iterator<Item = Self::Id>;
    /// Dimensions of point
    #[must_use]
    fn dimensions(&self) -> NonZeroUsize;
    /// Returns true of point id exists
    #[must_use]
    fn contains(&self, id: Self::Id) -> bool;
    /// Returns the dimension value of point `id`
    #[must_use]
    #[inline]
    fn coord(&self, id: Self::Id, dim: usize) -> Self::Value {
        self.get_coord(id, dim).expect("valid id and dim")
    }
    /// Returns the dimension value of point `id`, or `None` if `id` does not exist.
    #[must_use]
    fn get_coord(&self, id: Self::Id, dim: usize) -> Option<Self::Value>;
    /// Returns an iterator over the coords of `id`.
    #[must_use]
    #[inline]
    fn coords(&self, id: Self::Id) -> impl ExactSizeIterator<Item = &Self::Value> {
        self.get_coords(id).expect("valid id")
    }
    /// Returns an iterator over the coords of `id`, or `None` if `id` does not exist.
    #[must_use]
    fn get_coords(&self, id: Self::Id) -> Option<impl ExactSizeIterator<Item = &Self::Value>>;
    /// Returns the squared distance between `id` and `point`.
    #[must_use]
    #[inline]
    fn sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Self::Value {
        assert_eq!(
            point.len(),
            self.dimensions().get(),
            "point dimensions must match set dimension"
        );
        let mut sum = <Self::Value as ConstZero>::ZERO;
        for (d, &p) in point.iter().enumerate() {
            let diff = p - self.coord(id, d);
            sum += diff * diff;
        }
        sum
    }
    /// Returns the squared distance between `id` and `point`, or `None` if `id` does not exist or
    /// `point` does not match the collections dimensions.
    #[must_use]
    #[inline]
    fn get_sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Option<Self::Value> {
        if !self.contains(id) || point.len() != self.dimensions().get() {
            return None;
        }
        self.sq_distance(id, point).into()
    }
    /// Returns the squared distance between `id_a` and `id_b`.
    #[must_use]
    #[inline]
    fn sq_distance_between(&self, id_a: Self::Id, id_b: Self::Id) -> Self::Value {
        let mut sum = <Self::Value as ConstZero>::ZERO;
        for d in 0..self.dimensions().get() {
            let diff = self.coord(id_a, d) - self.coord(id_b, d);
            sum += diff * diff;
        }
        sum
    }
    /// Returns the squared distance between `id_a` and `id_b`, or `None` if `id_a` or `id_b` does
    /// not exist.
    #[must_use]
    #[inline]
    fn get_sq_distance_between(&self, id_a: Self::Id, id_b: Self::Id) -> Option<Self::Value> {
        if !self.contains(id_a) || !self.contains(id_b) {
            return None;
        }
        self.sq_distance_between(id_a, id_b).into()
    }
    /// Returns `id` as a boxed slice.
    #[must_use]
    #[inline]
    fn to_boxed_slice(&self, id: Self::Id) -> Option<Box<[Self::Value]>> {
        self.contains(id).then(|| {
            (0..self.dimensions().get())
                .map(|d| self.coord(id, d))
                .collect()
        })
    }
}
impl<P> PointSet for &P
where
    P: PointSet + ?Sized,
{
    type Id = P::Id;
    type Value = P::Value;
    #[inline]
    fn len(&self) -> NonZeroUsize { (**self).len() }
    #[inline]
    fn ids(&self) -> impl Iterator<Item = Self::Id> { (**self).ids() }
    #[inline]
    fn dimensions(&self) -> NonZeroUsize { (**self).dimensions() }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { (**self).contains(id) }
    #[inline]
    fn coord(&self, id: Self::Id, dim: usize) -> Self::Value { (**self).coord(id, dim) }
    #[inline]
    fn get_coord(&self, id: Self::Id, dim: usize) -> Option<Self::Value> {
        (**self).get_coord(id, dim)
    }
    #[inline]
    fn coords(&self, id: Self::Id) -> impl ExactSizeIterator<Item = &Self::Value> {
        (**self).coords(id)
    }
    #[inline]
    fn get_coords(&self, id: Self::Id) -> Option<impl ExactSizeIterator<Item = &Self::Value>> {
        (**self).get_coords(id)
    }
    #[inline]
    fn sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Self::Value {
        (**self).sq_distance(id, point)
    }
    #[inline]
    fn get_sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Option<Self::Value> {
        (**self).get_sq_distance(id, point)
    }
    #[inline]
    fn to_boxed_slice(&self, id: Self::Id) -> Option<Box<[Self::Value]>> {
        (**self).to_boxed_slice(id)
    }
}
