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

use std::num::NonZeroUsize;

pub use crate::number_traits::Number;

pub trait PointSet<N> {
    /// Number of points in set
    #[must_use]
    fn size(&self) -> NonZeroUsize;
    /// Iterator over point ids
    #[must_use]
    fn id_iter(&self) -> impl Iterator<Item = usize>;
    /// Dimensions of point
    #[must_use]
    fn dim(&self) -> NonZeroUsize;
    /// Returns true of point id exists
    #[must_use]
    fn exists(&self, id: usize) -> bool;
    /// Returns the dimension value of point `id`
    #[must_use]
    #[inline]
    fn coord(&self, id: usize, dim: usize) -> N
    where
        N: Copy,
    {
        self.try_coord(id, dim).expect("valid id and dim")
    }
    #[must_use]
    fn try_coord(&self, id: usize, dim: usize) -> Option<N>
    where
        N: Copy;
    #[must_use]
    #[inline]
    fn sq_distance(&self, id: usize, point: &[N]) -> N
    where
        N: Number,
    {
        assert_eq!(
            point.len(),
            self.dim().get(),
            "point dimensions must match set dimension"
        );
        let mut sum = N::zero();
        for (d, &p) in point.iter().enumerate() {
            let diff = p - self.coord(id, d);
            sum += diff * diff;
        }
        sum
    }
    #[must_use]
    #[inline]
    fn try_sq_distance(&self, id: usize, point: &[N]) -> Option<N>
    where
        N: Number,
    {
        if !self.exists(id) || point.len() != self.dim().get() {
            return None;
        }
        self.sq_distance(id, point).into()
    }
    #[must_use]
    #[inline]
    fn sq_distance_between(&self, id_a: usize, id_b: usize) -> N
    where
        N: Number,
    {
        let mut sum = N::zero();
        for d in 0..self.dim().get() {
            let diff = self.coord(id_a, d) - self.coord(id_b, d);
            sum += diff * diff;
        }
        sum
    }
    #[must_use]
    #[inline]
    fn try_sq_distance_between(&self, id_a: usize, id_b: usize) -> Option<N>
    where
        N: Number,
    {
        if !self.exists(id_a) || !self.exists(id_b) {
            return None;
        }
        self.sq_distance_between(id_a, id_b).into()
    }
    #[must_use]
    #[inline]
    fn to_boxed_slice(&self, id: usize) -> Option<Box<[N]>>
    where
        N: Copy,
    {
        self.exists(id)
            .then(|| (0..self.dim().get()).map(|d| self.coord(id, d)).collect())
    }
}
impl<N, P> PointSet<N> for &P
where
    P: PointSet<N> + ?Sized,
{
    #[inline]
    fn size(&self) -> NonZeroUsize { (**self).size() }
    #[inline]
    fn id_iter(&self) -> impl Iterator<Item = usize> { (**self).id_iter() }
    #[inline]
    fn dim(&self) -> NonZeroUsize { (**self).dim() }
    #[inline]
    fn exists(&self, id: usize) -> bool { (**self).exists(id) }
    #[inline]
    fn coord(&self, id: usize, dim: usize) -> N
    where
        N: Copy,
    {
        (**self).coord(id, dim)
    }
    #[inline]
    fn try_coord(&self, id: usize, dim: usize) -> Option<N>
    where
        N: Copy,
    {
        (**self).try_coord(id, dim)
    }
    #[inline]
    fn sq_distance(&self, id: usize, point: &[N]) -> N
    where
        N: Number,
    {
        (**self).sq_distance(id, point)
    }
    #[inline]
    fn try_sq_distance(&self, id: usize, point: &[N]) -> Option<N>
    where
        N: Number,
    {
        (**self).try_sq_distance(id, point)
    }
    #[inline]
    fn to_boxed_slice(&self, id: usize) -> Option<Box<[N]>>
    where
        N: Copy,
    {
        (**self).to_boxed_slice(id)
    }
}
