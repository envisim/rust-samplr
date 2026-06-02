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

use crate::number_traits::Number;

pub trait PointSet {
    type Id: Sized + Eq + Hash + Ord + Copy + Debug;
    type Value: Number;

    /// Number of points in set
    #[must_use]
    fn size(&self) -> NonZeroUsize;
    /// Iterator over point ids
    #[must_use]
    fn id_iter(&self) -> impl Iterator<Item = Self::Id>;
    /// Dimensions of point
    #[must_use]
    fn dim(&self) -> NonZeroUsize;
    /// Returns true of point id exists
    #[must_use]
    fn exists(&self, id: Self::Id) -> bool;
    /// Returns the dimension value of point `id`
    #[must_use]
    #[inline]
    fn coord(&self, id: Self::Id, dim: usize) -> Self::Value {
        self.try_coord(id, dim).expect("valid id and dim")
    }
    #[must_use]
    fn try_coord(&self, id: Self::Id, dim: usize) -> Option<Self::Value>;
    #[must_use]
    #[inline]
    fn sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Self::Value {
        assert_eq!(
            point.len(),
            self.dim().get(),
            "point dimensions must match set dimension"
        );
        let mut sum = <Self::Value as ConstZero>::ZERO;
        for (d, &p) in point.iter().enumerate() {
            let diff = p - self.coord(id, d);
            sum += diff * diff;
        }
        sum
    }
    #[must_use]
    #[inline]
    fn try_sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Option<Self::Value> {
        if !self.exists(id) || point.len() != self.dim().get() {
            return None;
        }
        self.sq_distance(id, point).into()
    }
    #[must_use]
    #[inline]
    fn sq_distance_between(&self, id_a: Self::Id, id_b: Self::Id) -> Self::Value {
        let mut sum = <Self::Value as ConstZero>::ZERO;
        for d in 0..self.dim().get() {
            let diff = self.coord(id_a, d) - self.coord(id_b, d);
            sum += diff * diff;
        }
        sum
    }
    #[must_use]
    #[inline]
    fn try_sq_distance_between(&self, id_a: Self::Id, id_b: Self::Id) -> Option<Self::Value> {
        if !self.exists(id_a) || !self.exists(id_b) {
            return None;
        }
        self.sq_distance_between(id_a, id_b).into()
    }
    #[must_use]
    #[inline]
    fn to_boxed_slice(&self, id: Self::Id) -> Option<Box<[Self::Value]>> {
        self.exists(id)
            .then(|| (0..self.dim().get()).map(|d| self.coord(id, d)).collect())
    }
}
impl<P> PointSet for &P
where
    P: PointSet + ?Sized,
{
    type Id = P::Id;
    type Value = P::Value;
    #[inline]
    fn size(&self) -> NonZeroUsize { (**self).size() }
    #[inline]
    fn id_iter(&self) -> impl Iterator<Item = Self::Id> { (**self).id_iter() }
    #[inline]
    fn dim(&self) -> NonZeroUsize { (**self).dim() }
    #[inline]
    fn exists(&self, id: Self::Id) -> bool { (**self).exists(id) }
    #[inline]
    fn coord(&self, id: Self::Id, dim: usize) -> Self::Value { (**self).coord(id, dim) }
    #[inline]
    fn try_coord(&self, id: Self::Id, dim: usize) -> Option<Self::Value> {
        (**self).try_coord(id, dim)
    }
    #[inline]
    fn sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Self::Value {
        (**self).sq_distance(id, point)
    }
    #[inline]
    fn try_sq_distance(&self, id: Self::Id, point: &[Self::Value]) -> Option<Self::Value> {
        (**self).try_sq_distance(id, point)
    }
    #[inline]
    fn to_boxed_slice(&self, id: Self::Id) -> Option<Box<[Self::Value]>> {
        (**self).to_boxed_slice(id)
    }
}
