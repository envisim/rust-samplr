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

use crate::number_traits::Number;
use crate::spatial::PointSet;

#[derive(Clone, Debug, Copy)]
pub struct Split<N> {
    dimension: usize,
    value: N,
    /// If `true`, equal values goes to the left
    leq: bool,
}
impl<N> Split<N> {
    #[inline]
    pub fn dimension(&self) -> usize { self.dimension }
    #[inline]
    pub fn value(&self) -> N
    where
        N: Copy,
    {
        self.value
    }
    #[inline]
    pub fn leq(&self) -> bool { self.leq }
    #[inline]
    pub fn is_left(&self, value: N) -> bool
    where
        N: Number,
    {
        value < self.value || (self.leq && value == self.value)
    }
    #[inline]
    pub fn unit_is_left(&self, unit: &[N]) -> bool
    where
        N: Number,
    {
        self.is_left(unit[self.dimension])
    }
    #[inline]
    pub fn abs_distance(&self, value: N) -> (bool, N)
    where
        N: Number,
    {
        if self.is_left(value) {
            (true, self.value - value)
        } else {
            (false, value - self.value)
        }
    }
    #[inline]
    pub fn unit_abs_distance(&self, unit: &[N]) -> (bool, N)
    where
        N: Number,
    {
        self.abs_distance(unit[self.dimension])
    }
}
pub struct SplitUnit<N> {
    split: Split<N>,
    /// First unit to the right of the split. Must be in (0, len)
    unit: usize,
}
impl<N> SplitUnit<N> {
    #[inline]
    pub fn new(split: Split<N>, unit: usize) -> Self { Self { split, unit } }
    #[inline]
    pub fn split(&self) -> &Split<N> { &self.split }
    #[inline]
    pub fn dimension(&self) -> usize { self.split.dimension }
    #[inline]
    pub fn value(&self) -> N
    where
        N: Copy,
    {
        self.split.value
    }
    #[inline]
    pub fn leq(&self) -> bool { self.split.leq }
    #[inline]
    pub fn unit(&self) -> usize { self.unit }
    #[inline]
    fn set_unit<T>(&mut self, data: &T, units: &mut [usize])
    where
        N: Number,
        T: PointSet<N>,
    {
        let mut left: usize = 0;
        let mut right: usize = units.len();
        while left < right {
            let v = data.coord(units[left], self.dimension());
            if self.split.is_left(v) {
                left += 1;
            } else {
                right -= 1;
                units.swap(left, right);
            }
        }
        self.unit = left;
    }
}
impl<N> From<SplitUnit<N>> for Split<N> {
    fn from(su: SplitUnit<N>) -> Self { su.split }
}

#[derive(Clone, Copy, Debug)]
pub struct Border<N> {
    min: N,
    max: N,
}
impl<N> Border<N> {
    #[inline]
    fn min(&self) -> N
    where
        N: Copy,
    {
        self.min
    }
    #[inline]
    fn max(&self) -> N
    where
        N: Copy,
    {
        self.max
    }
    #[inline]
    fn set_min(&mut self, v: N)
    where
        N: Number,
    {
        if v < self.min {
            self.min = v;
        }
    }
    #[inline]
    fn set_max(&mut self, v: N)
    where
        N: Number,
    {
        if self.max < v {
            self.max = v;
        }
    }
    #[inline]
    fn range(&self) -> N
    where
        N: Number,
    {
        self.max - self.min
    }
    #[inline]
    fn centre(&self) -> N
    where
        N: Number,
    {
        self.min.mid(self.max)
    }
    /// Returns the centre if it is possible to place in (min, max), including some checks for
    /// seeing if max-min is large enough.
    #[inline]
    fn valid_centre(&self) -> Option<N>
    where
        N: Number,
    {
        let range = self.range();
        if range <= N::epsilonish() || !range.is_finite() {
            return None;
        }
        let centre = self.centre();
        if !centre.is_finite() {
            return None;
        }
        // Also reject when min/max are too close to distinguish
        let scale = if self.min.abs() < self.max.abs() {
            self.max.abs()
        } else {
            self.min.abs()
        };
        // Consider larger epsilon?
        (scale * N::epsilonish() < range).then_some(centre)
    }
    #[inline]
    fn from_data<T>(data: &T, units: &[usize], dim: usize) -> Self
    where
        N: Number,
        T: PointSet<N>,
    {
        if units.is_empty() {
            return Self::default();
        }
        let mut b = Self {
            min: data.coord(0, dim),
            max: data.coord(0, dim),
        };
        for &id in units.iter().skip(1) {
            b.set_min(data.coord(id, dim));
            b.set_max(data.coord(id, dim));
        }
        b
    }
}
impl<N> Default for Border<N>
where
    N: Number,
{
    fn default() -> Self {
        Self {
            min: N::zero(),
            max: N::zero(),
        }
    }
}

pub trait FindSplit<N, T>
where
    Self: Sized,
    T: PointSet<N>,
{
    fn split(self, data: &T, units: &mut [usize]) -> Option<(SplitUnit<N>, Self, Self)>
    where
        N: Number;
}

impl<N, T> FindSplit<N, T> for MidpointSlide<N>
where
    Self: Sized,
    T: PointSet<N>,
{
    #[inline]
    fn split(mut self, data: &T, units: &mut [usize]) -> Option<(SplitUnit<N>, Self, Self)>
    where
        N: Number,
    {
        let split = self.find_split(data, units)?;
        let mut left = self.clone();
        let mut right = self;
        left.borders[split.dimension()].max = split.value();
        right.borders[split.dimension()].min = split.value();
        Some((split, left, right))
    }
}

#[derive(Clone, Debug)]
pub struct MidpointSlide<N> {
    borders: Box<[Border<N>]>,
}
impl<N> MidpointSlide<N> {
    #[inline]
    pub fn new<T>(data: &T, units: &[usize]) -> Self
    where
        N: Number,
        T: PointSet<N>,
    {
        Self {
            borders: (0..data.dim().get())
                .map(|d| Border::from_data(data, units, d))
                .collect(),
        }
    }
    #[inline]
    pub fn get(&self, dim: usize) -> Option<&Border<N>> { self.borders.get(dim) }
    /// Sort dims by range
    #[inline]
    fn order(&self) -> Box<[usize]>
    where
        N: Number,
    {
        let mut indices: Box<[usize]> = (0..self.borders.len()).collect();
        indices
            .sort_unstable_by(|&a, &b| self.borders[b].range().compare(&self.borders[a].range()));
        indices
    }
    /// Redraw borders for a dimension. Returns `true` if borders are not degenerate
    #[inline]
    fn redraw<T>(&mut self, dim: usize, data: &T, units: &[usize]) -> bool
    where
        N: Number,
        T: PointSet<N>,
    {
        self.borders[dim] = Border::from_data(data, units, dim);
        self.borders[dim].min() != self.borders[dim].max()
    }
    /// The midpoint slide splitting method.
    /// Returns a split, where units `[0..unit)` have values < `value`, and units [unit,..) have
    /// values > `value`.
    /// If `leq` is `true`, the first group also contains equal elements, otherwise the right group
    /// contains equal elements.
    ///
    /// Returns `None` if no such split exists
    ///
    /// # References
    /// Maneewongvatana, S., & Mount, D. M. (1999).
    /// It’s okay to be skinny, if your friends are fat.
    /// In Center for geometric computing 4th annual workshop on computational geometry (Vol. 2).
    fn find_split<T>(&mut self, data: &T, units: &mut [usize]) -> Option<SplitUnit<N>>
    where
        N: Number,
        T: PointSet<N>,
    {
        assert_eq!(data.dim().get(), self.borders.len());

        if units.is_empty() {
            return None;
        }

        let mut split = SplitUnit {
            split: Split {
                dimension: 0,
                value: N::zero(),
                leq: true,
            },
            unit: 0,
        };

        // Sort dims by range
        for &dim in self.order().iter() {
            split.split.dimension = dim;

            // If the current border is degenerate, we assume any subsequent borders to be degenerate as
            // well -- as indices is sorted by border range -- and do an early return.
            let centre = self.borders[dim].valid_centre()?;

            split.split.value = centre;
            split.split.leq = true;
            split.set_unit(data, units);

            if split.unit() == 0 || split.unit() == units.len() {
                // If the split value does not split the units, the borders might not be good
                // anymore, so we should redraw them.
                if !self.redraw(dim, data, units) {
                    // If the redrawn borders are degenerate, we should look at the next border in
                    // the list.
                    continue;
                }

                // Otherwise, we now know that the new borders touches some units. Depending on the
                // direction of the first split, we use the new borders as the new split.
                if split.unit() == 0 {
                    // All units were above the last split value. Hence, we could split by the
                    // min border.
                    split.split.value = self.borders[dim].min();
                    // split.split.leq = true;
                } else {
                    // All units were below the last split value. Hence, we could split by the
                    // max border.
                    split.split.value = self.borders[dim].max();
                    split.split.leq = false;
                }

                split.set_unit(data, units);

                // We still failed, somehow ... giving up
                if split.unit() == 0 || split.unit() == units.len() {
                    return None;
                }
            }

            return Some(split);
        }

        // All borders are degenerate
        None
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::PointSet;
    use crate::kd_tree::split_methods::{
        Border,
        Split,
        midpoint_slide,
    };
    use crate::matrix::Matrix;

    fn mat_from(data: Vec<f64>, rows: usize) -> Matrix<'static> {
        Matrix::from_vec(data, NonZeroUsize::new(rows).unwrap()).unwrap()
    }

    // --- Border ---

    #[test]
    fn border_range_and_centre() {
        let b = Border { min: 2.0, max: 6.0 };
        assert_eq!(b.range(), 4.0);
        assert_eq!(b.centre(), 4.0);
    }

    #[test]
    fn border_valid_centre_rejects_degenerate_range() {
        let b = Border { min: 1.0, max: 1.0 };
        assert!(b.valid_centre().is_none());
    }

    #[test]
    fn border_valid_centre_rejects_non_finite() {
        let b = Border {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        };
        assert!(b.valid_centre().is_none());
    }

    #[test]
    fn border_valid_centre_accepts_normal_range() {
        let b = Border {
            min: 0.0,
            max: 10.0,
        };
        assert_eq!(b.valid_centre(), Some(5.0));
    }

    #[test]
    fn border_from_data_computes_min_max() {
        // 4 points in 2D: (0,10), (5,20), (3,15), (7,5)
        let m = mat_from(vec![0.0, 5.0, 3.0, 7.0, 10.0, 20.0, 15.0, 5.0], 4);
        let units: Vec<usize> = (0..4).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        assert_eq!(borders.len(), 2);
        assert_eq!(borders[0].min, 0.0);
        assert_eq!(borders[0].max, 7.0);
        assert_eq!(borders[1].min, 5.0);
        assert_eq!(borders[1].max, 20.0);
    }

    #[test]
    fn border_from_data_handles_empty_units() {
        let m = mat_from(vec![0.0, 5.0, 10.0, 20.0], 2);
        let borders = Border::from_data_to_vec(&m, &[]);
        // Default Border { min: 0.0, max: 0.0 }
        for b in borders.iter() {
            assert_eq!(b.min, 0.0);
            assert_eq!(b.max, 0.0);
        }
    }

    // --- Split ---

    #[test]
    fn split_value_is_left_with_leq_true() {
        // We can't construct Split directly since fields are private,
        // so we exercise it via midpoint_slide. See below.
        let m = mat_from(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], 4);
        let mut units: Vec<usize> = (0..4).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        let sp = midpoint_slide(&m, &borders, &mut units).expect("split exists");
        let split: Split = sp.into();

        // Dimension 0 has range 3 (vs dim 1 range 0), so split is on dim 0 at centre 1.5
        assert_eq!(split.dimension(), 0);
        assert_eq!(split.value(), 1.5);
        // With leq=true (default for midpoint_slide), 1.5 itself goes left
        assert!(split.is_left(1.5));
        assert!(split.is_left(1.0));
        assert!(!split.is_left(2.0));
    }

    // --- midpoint_slide ---

    #[test]
    fn midpoint_slide_splits_on_widest_dim() {
        // dim 0 range = 10, dim 1 range = 2 -> split on dim 0
        let m = mat_from(vec![0.0, 5.0, 10.0, 1.0, 2.0, 3.0], 3);
        let mut units: Vec<usize> = (0..3).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        let sp = midpoint_slide(&m, &borders, &mut units).expect("split exists");
        assert_eq!(sp.dimension(), 0);
        // Range is 10, centre is 5.0. With leq=true, the point at 5.0 goes left.
        assert_eq!(sp.value(), 5.0);
        // Unit id is first unit to the RIGHT of split. Points 0 and 5 are left,
        // point 10 is right. So unit == 2 (index in the reordered slice).
        assert_eq!(sp.unit(), 2);
    }

    #[test]
    fn midpoint_slide_returns_none_for_degenerate_data() {
        // All points coincide -> no valid split on any dim
        let m = mat_from(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 3);
        let mut units: Vec<usize> = (0..3).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        assert!(midpoint_slide(&m, &borders, &mut units).is_none());
    }

    #[test]
    fn midpoint_slide_returns_none_for_empty_units() {
        let m = mat_from(vec![0.0, 5.0, 10.0, 1.0, 2.0, 3.0], 3);
        let borders = Border::from_data_to_vec(&m, &[]);
        let mut empty: Vec<usize> = vec![];
        assert!(midpoint_slide(&m, &borders, &mut empty).is_none());
    }

    #[test]
    fn midpoint_slide_partitions_units_correctly() {
        // Spread 6 points on dim 0 from 0..5; dim 1 is arbitrary
        // dim 0: 0, 1, 2, 3, 4, 5 ; centre = 2.5 -> 0..3 left, 3..6 right
        let m = mat_from(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, // dim 0
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // dim 1
            ],
            6,
        );
        let mut units: Vec<usize> = (0..6).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        let sp = midpoint_slide(&m, &borders, &mut units).expect("split exists");
        let split = sp.split();
        let unit = sp.unit();

        assert_eq!(split.dimension(), 0);
        assert_eq!(split.value(), 2.5);

        // Every unit in 0..unit should be left of split, every unit in unit.. right
        for (i, &u) in units.iter().enumerate() {
            let c = m.coord(u, 0);
            // units (i) to the left of unit should be to the left according to split
            assert!((i < unit) == split.value_is_left(c));
        }
    }

    #[test]
    fn midpoint_slide_handles_cluster_on_one_side() {
        // Cluster at 0.0 and one outlier at 10.0
        // centre = 5.0; naive split puts all zeros left, one right
        let m = mat_from(
            vec![
                0.0, 0.0, 0.0, 0.0, 10.0, // dim 0
                0.0, 0.0, 0.0, 0.0, 0.0, // dim 1
            ],
            5,
        );
        let mut units: Vec<usize> = (0..5).collect();
        let borders = Border::from_data_to_vec(&m, &units);
        let sp = midpoint_slide(&m, &borders, &mut units).expect("split exists");
        // Split produces a non-empty left and non-empty right
        assert!(0 < sp.unit() && sp.unit() < units.len());
    }
}
