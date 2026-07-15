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

use super::Number;

/// Methods for accessing containers of points in some coordinate system.
/// The container must not be empty.
#[expect(clippy::len_without_is_empty, reason = "cannot be empty")]
pub trait PointSet {
    /// The type of the identifiers
    type Id: Sized + Eq + Hash + Ord + Copy + Debug;
    /// The type of the values
    type Value: Number;

    /// Number of points in set
    #[must_use]
    fn len(&self) -> NonZeroUsize;
    /// Iterator over point ids
    #[must_use]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone;
    /// Dimensions of point
    #[must_use]
    fn dimensions(&self) -> NonZeroUsize;
    /// Returns true of point id exists
    #[must_use]
    fn contains(&self, id: Self::Id) -> bool;
    /// Returns the dimension value of point `id`, or `None` if `id` does not exist.
    #[must_use]
    fn coord(&self, id: Self::Id, dim: usize) -> Option<&Self::Value>;
    /// Returns the dimension value of point `id`
    /// # Safety
    /// The `id` or `dim` is not checked, and may be out of bounds.
    #[must_use]
    #[inline]
    unsafe fn coord_unchecked(&self, id: Self::Id, dim: usize) -> &Self::Value {
        self.coord(id, dim).expect("valid id and dim")
    }
    /// Returns an iterator over the coords of `id`, or `None` if `id` does not exist.
    #[must_use]
    fn coords(
        &self,
        id: Self::Id,
    ) -> Option<impl ExactSizeIterator<Item = &Self::Value> + DoubleEndedIterator + Clone>;
    /// Returns an iterator over the coordinates of each id in the set
    #[must_use]
    #[inline]
    fn iter(
        &self,
    ) -> impl ExactSizeIterator<
        Item = impl ExactSizeIterator<Item = (Self::Id, usize, &Self::Value)>
               + DoubleEndedIterator
               + Clone,
    > + Clone {
        self.ids().map(move |id| {
            (0..self.dimensions().get()).map(move |c| {
                // SAFETY:
                // id and dimensions come from the trait impl
                (id, c, unsafe { self.coord_unchecked(id, c) })
            })
        })
    }
    /// Returns an iterator over the ids of each dimension in the set
    #[must_use]
    #[inline]
    fn columns(
        &self,
    ) -> impl ExactSizeIterator<
        Item = impl ExactSizeIterator<Item = (Self::Id, usize, &Self::Value)> + Clone,
    > + DoubleEndedIterator
    + Clone {
        (0..self.dimensions().get()).map(move |c| {
            self.ids().map(move |id| {
                // SAFETY:
                // id and dimensions come from the trait impl
                (id, c, unsafe { self.coord_unchecked(id, c) })
            })
        })
    }
    /// Returns the squared distance between `id` and `point`, or `None` if `id` does not exist or
    /// `point` does not match the collections dimensions.
    #[must_use]
    #[inline]
    fn sq_distance<'bitem, P, ITER>(&self, id: Self::Id, point: P) -> Option<Self::Value>
    where
        P: IntoIterator<Item = &'bitem Self::Value, IntoIter = ITER>,
        ITER: ExactSizeIterator<Item = &'bitem Self::Value>,
        Self: 'bitem,
    {
        let point = point.into_iter();
        if point.len() != self.dimensions().get() {
            return None;
        }
        Some(
            self.coords(id)?
                .zip(point)
                .map(|(a, b)| {
                    let diff = *a - *b;
                    diff * diff
                })
                .sum(),
        )
    }
    /// Returns the squared distance between `id` and `point`.
    /// # Safety
    /// The `id` or `point` dimension is not checked, and may be out of bounds.
    #[must_use]
    #[inline]
    unsafe fn sq_distance_unchecked<'bitem, P>(&self, id: Self::Id, point: P) -> Self::Value
    where
        P: IntoIterator<Item = &'bitem Self::Value>,
        Self: 'bitem,
    {
        let point = point.into_iter();
        self.coords(id)
            .expect("valid id")
            .zip(point)
            .map(|(a, b)| {
                let diff = *a - *b;
                diff * diff
            })
            .sum()
    }
    /// Returns the squared distance between `id_a` and `id_b`, or `None` if `id_a` or `id_b` does
    /// not exist.
    #[must_use]
    #[inline]
    fn sq_distance_between(&self, id_a: Self::Id, id_b: Self::Id) -> Option<Self::Value> {
        Some(
            self.coords(id_a)?
                .zip(self.coords(id_b)?)
                .map(|(a, b)| {
                    let diff = *a - *b;
                    diff * diff
                })
                .sum(),
        )
    }
}

impl<PS> PointSet for &PS
where
    PS: PointSet + ?Sized,
{
    type Id = PS::Id;
    type Value = PS::Value;
    #[inline]
    fn len(&self) -> NonZeroUsize { (**self).len() }
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { (**self).ids() }
    #[inline]
    fn dimensions(&self) -> NonZeroUsize { (**self).dimensions() }
    #[inline]
    fn contains(&self, id: Self::Id) -> bool { (**self).contains(id) }
    #[inline]
    fn coord(&self, id: Self::Id, dim: usize) -> Option<&Self::Value> { (**self).coord(id, dim) }
    #[inline]
    unsafe fn coord_unchecked(&self, id: Self::Id, dim: usize) -> &Self::Value {
        // SAFETY:
        // See Safety-section on `PointSet`
        unsafe { (**self).coord_unchecked(id, dim) }
    }
    #[inline]
    fn coords(
        &self,
        id: Self::Id,
    ) -> Option<impl ExactSizeIterator<Item = &Self::Value> + DoubleEndedIterator + Clone> {
        (**self).coords(id)
    }
    #[inline]
    fn sq_distance<'bitem, P, ITER>(&self, id: Self::Id, point: P) -> Option<Self::Value>
    where
        P: IntoIterator<Item = &'bitem Self::Value, IntoIter = ITER>,
        ITER: ExactSizeIterator<Item = &'bitem Self::Value>,
        Self: 'bitem,
    {
        (**self).sq_distance(id, point)
    }
    #[inline]
    unsafe fn sq_distance_unchecked<'bitem, P>(&self, id: Self::Id, point: P) -> Self::Value
    where
        P: IntoIterator<Item = &'bitem Self::Value>,
        Self: 'bitem,
    {
        // SAFETY:
        // See Safety-section on `PointSet`
        unsafe { (**self).sq_distance_unchecked(id, point) }
    }
}
