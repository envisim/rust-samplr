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

//! Split methods
//!
//! A split method defines a splitting strategy for a kd-tree.

pub use split::{
    Split,
    SplitUnit,
};

use crate::number_traits::Number;
use crate::spatial::PointSet;

mod split {
    //! Defines a split, and a split with a unit

    use crate::number_traits::Number;
    use crate::spatial::PointSet;

    /// Defines a split in a tree [`Branch`].
    #[must_use]
    #[derive(Clone, Debug, Copy)]
    pub struct Split<N> {
        /// The dimension of the split.
        pub dimension: usize,
        /// The value of the split.
        pub value: N,
        /// If `true`, equal values goes to the left.
        pub leq: bool,
    }
    impl<N> Split<N> {
        /// Constructs a new split
        #[inline]
        pub fn new(dimension: usize, value: N, leq: bool) -> Self {
            Self {
                dimension,
                value,
                leq,
            }
        }
        /// Returns `true` if a value is to the left of the split.
        #[must_use]
        #[inline]
        pub fn is_left(&self, value: N) -> bool
        where
            N: Number,
        {
            value < self.value || (self.leq && value == self.value)
        }
        /// Returns `true` if a unit is left of the split.
        /// Panics if the split dimension is oob of the unit.
        #[must_use]
        #[inline]
        pub fn unit_is_left(&self, unit: &[N]) -> bool
        where
            N: Number,
        {
            self.is_left(unit[self.dimension])
        }
        /// Calculates the absolute distance to the split in the dimension of the split.
        #[must_use]
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
        /// Calculates the absolute distance between a unit and the split in the dimension of the split.
        #[must_use]
        #[inline]
        pub fn unit_abs_distance(&self, unit: &[N]) -> (bool, N)
        where
            N: Number,
        {
            self.abs_distance(unit[self.dimension])
        }
    }

    /// Defines a split, and the first unit to the right of the split.
    /// Used as the return value of the [`FindSplit`] trait.
    #[must_use]
    pub struct SplitUnit<N> {
        /// The split.
        pub split: Split<N>,
        /// First unit to the right of the split. Must be in (0, len).
        pub unit: usize,
    }
    impl<N> SplitUnit<N> {
        /// Constructs a new split
        #[inline]
        pub fn new(dimension: usize, value: N, leq: bool, unit: usize) -> Self {
            Self {
                split: Split {
                    dimension,
                    value,
                    leq,
                },
                unit,
            }
        }
        /// Sorts units according to the split, and sets unit so that it represents the index of the
        /// first unit to the right.
        #[inline]
        pub(super) fn set_unit<T>(&mut self, data: &T, units: &mut [usize])
        where
            N: Number,
            T: PointSet<N = N>,
        {
            let mut left: usize = 0;
            let mut right: usize = units.len();
            while left < right {
                let v = data.coord(units[left], self.split.dimension);
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
        #[inline]
        fn from(su: SplitUnit<N>) -> Self { su.split }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_split_is_left() {
            // Case 1: leq = true (Less than or equal goes left)
            let split_leq = Split::new(0, 10.0, true);
            assert!(split_leq.is_left(5.0)); // 5 < 10
            assert!(split_leq.is_left(10.0)); // 10 == 10
            assert!(!split_leq.is_left(15.0)); // 15 > 10

            // Case 2: leq = false (Strictly less than goes left)
            let split_strict = Split::new(0, 10.0, false);
            assert!(split_strict.is_left(5.0)); // 5 < 10
            assert!(!split_strict.is_left(10.0)); // 10 == 10 (goes right)
            assert!(!split_strict.is_left(15.0)); // 15 > 10
        }

        #[test]
        fn test_split_unit_is_left() {
            let split = Split::new(1, 5.0, true); // Splitting on dimension 1
            let unit = vec![0.0, 3.0, 10.0]; // Dim 1 value is 3.0
            assert!(split.unit_is_left(&unit));

            let unit_right = vec![0.0, 7.0, 10.0]; // Dim 1 value is 7.0
            assert!(!split.unit_is_left(&unit_right));
        }

        #[test]
        #[should_panic]
        fn test_split_unit_oob_panic() {
            let split = Split::new(5, 10.0, true);
            let unit = vec![1.0, 2.0]; // Dimension 5 is out of bounds
            let _ = split.unit_is_left(&unit);
        }

        #[test]
        fn test_split_abs_distance() {
            let split = Split::new(0, 10.0, true);

            // (is_left, distance)
            assert_eq!(split.abs_distance(12.0), (false, 2.0));
            assert_eq!(split.abs_distance(7.0), (true, 3.0));
            assert_eq!(split.abs_distance(10.0), (true, 0.0));
        }

        #[test]
        fn test_split_unit_abs_distance() {
            let split = Split::new(0, 10.0, true);
            let unit = vec![15.0, 0.0];
            assert_eq!(split.unit_abs_distance(&unit), (false, 5.0));
        }

        #[test]
        fn test_split_unit_conversion() {
            let su = SplitUnit::new(0, 5.0, true, 42);
            let s: Split<f64> = su.into();
            assert_eq!(s.dimension, 0);
            assert_eq!(s.value, 5.0);
            assert!(s.leq);
        }
    }
}

/// Represents the border of a tree window
#[must_use]
#[derive(Clone, Copy, Debug)]
struct Border<N> {
    /// The minimum (left) border
    min: N,
    /// The maximum (right) border
    max: N,
}
impl<N> Border<N> {
    /// Sets `min` to `v` if `v < min`
    #[inline]
    fn set_min(&mut self, v: N)
    where
        N: Number,
    {
        if v < self.min {
            self.min = v;
        }
    }
    /// Sets `max` to `v` if `v > max`
    #[inline]
    fn set_max(&mut self, v: N)
    where
        N: Number,
    {
        if self.max < v {
            self.max = v;
        }
    }
    /// Returns the width of the border.
    #[must_use]
    #[inline]
    fn range(&self) -> N
    where
        N: Number,
    {
        self.max - self.min
    }
    /// Returns the midpoint of the border
    #[must_use]
    #[inline]
    fn centre(&self) -> N
    where
        N: Number,
    {
        self.min.mid(self.max)
    }
    /// Returns the centre if it is possible to place in (min, max), including some checks for
    /// seeing if max-min is large enough.
    #[must_use]
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
    /// Constructs a new border from the `units` according to `data` in a certain dimension `dim`.
    #[inline]
    fn from_data<T>(data: &T, units: &[usize], dim: usize) -> Self
    where
        N: Number,
        T: PointSet<N = N>,
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
    #[inline]
    fn default() -> Self {
        Self {
            min: N::zero(),
            max: N::zero(),
        }
    }
}

/// Splits the data into left and right according to some algorithm.
pub trait FindSplit<T>
where
    Self: Sized,
    T: PointSet,
{
    /// Returns the split as:
    /// `SplitUnit`, the split and index of first right-unit.
    /// The left and right splits.
    fn split(self, data: &T, units: &mut [usize]) -> Option<(SplitUnit<T::N>, Self, Self)>;
}

impl<T> FindSplit<T> for MidpointSlide<T::N>
where
    Self: Sized,
    T: PointSet,
{
    #[must_use]
    #[inline]
    fn split(mut self, data: &T, units: &mut [usize]) -> Option<(SplitUnit<T::N>, Self, Self)> {
        let split = self.find_split(data, units)?;
        let mut left = self.clone();
        let mut right = self;
        left.borders[split.split.dimension].max = split.split.value;
        right.borders[split.split.dimension].min = split.split.value;
        Some((split, left, right))
    }
}

/// The midpoint slide splitting method.
///
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
#[must_use]
#[derive(Clone, Debug)]
pub struct MidpointSlide<N> {
    /// The borders for all dimensions, one is potentially to be split
    borders: Box<[Border<N>]>,
}
impl<N> MidpointSlide<N> {
    /// Constructs a new base window from the `units` according to `data`.
    #[inline]
    pub fn new<T>(data: &T, units: &[usize]) -> Self
    where
        N: Number,
        T: PointSet<N = N>,
    {
        Self {
            borders: (0..data.dim().get())
                .map(|d| Border::from_data(data, units, d))
                .collect(),
        }
    }
    /// Sort dims by range.
    #[must_use]
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
    #[must_use]
    #[inline]
    fn redraw<T>(&mut self, dim: usize, data: &T, units: &[usize]) -> bool
    where
        N: Number,
        T: PointSet<N = N>,
    {
        self.borders[dim] = Border::from_data(data, units, dim);
        self.borders[dim].min != self.borders[dim].max
    }
    /// Finds the split according to the midpoint slide splitting method.
    ///
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
    ///
    /// # Panics
    /// Will panic if data dimensions does not match the number of borders in the split.
    /// This implies that the split is run on different data than for the previous split.
    #[must_use]
    fn find_split<T>(&mut self, data: &T, units: &mut [usize]) -> Option<SplitUnit<N>>
    where
        N: Number,
        T: PointSet<N = N>,
    {
        assert_eq!(
            data.dim().get(),
            self.borders.len(),
            "data dimensions must match the size of the number of borders in the split"
        );

        if units.is_empty() {
            return None;
        }

        let mut split = SplitUnit::new(0, N::zero(), true, 0);

        // Sort dims by range
        for &dim in &self.order() {
            split.split.dimension = dim;

            // If the current border is degenerate, we assume any subsequent borders to be degenerate as
            // well -- as indices is sorted by border range -- and do an early return.
            let centre = self.borders[dim].valid_centre()?;

            split.split.value = centre;
            split.split.leq = true;
            split.set_unit(data, units);

            if split.unit == 0 || split.unit == units.len() {
                // If the split value does not split the units, the borders might not be good
                // anymore, so we should redraw them.
                if !self.redraw(dim, data, units) {
                    // If the redrawn borders are degenerate, we should look at the next border in
                    // the list.
                    continue;
                }

                // Otherwise, we now know that the new borders touches some units. Depending on the
                // direction of the first split, we use the new borders as the new split.
                if split.unit == 0 {
                    // All units were above the last split value. Hence, we could split by the
                    // min border.
                    split.split.value = self.borders[dim].min;
                    // split.split.leq = true;
                } else {
                    // All units were below the last split value. Hence, we could split by the
                    // max border.
                    split.split.value = self.borders[dim].max;
                    split.split.leq = false;
                }

                split.set_unit(data, units);

                // We still failed, somehow ... giving up
                if split.unit == 0 || split.unit == units.len() {
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
    use super::*;
    use crate::matrix::Matrix;
    use crate::test_utils::*;

    /// Helper to create a NonZeroUsize

    /// Creates a 2D Matrix PointSet for testing
    /// Layout: Column-major (all dimension 0 values, then all dimension 1 values)
    fn setup_test_matrix() -> Matrix<f64> {
        let data = vec![
            0.0, 10.0, // Dim 0 (X)
            0.0, 10.0, // Dim 1 (Y)
        ];
        // 2 points, 2 dimensions
        Matrix::new(data, nz(2)).unwrap()
    }

    #[test]
    fn test_border_properties() {
        let mut border = Border {
            min: 0.0,
            max: 10.0,
        };

        assert_eq!(border.range(), 10.0);
        assert_eq!(border.centre(), 5.0);
        assert_eq!(border.valid_centre(), Some(5.0));

        border.set_min(-1.0);
        border.set_max(11.0);
        assert_eq!(border.min, -1.0);
        assert_eq!(border.max, 11.0);
    }

    #[test]
    fn test_border_from_data() {
        let mat = setup_test_matrix(); // Points: (0,0) and (10,10)
        let units = vec![0, 1];

        let border_dim0 = Border::from_data(&mat, &units, 0);
        assert_eq!(border_dim0.min, 0.0);
        assert_eq!(border_dim0.max, 10.0);
    }

    #[test]
    fn test_midpoint_slide_construction() {
        let mat = setup_test_matrix();
        let units = vec![0, 1];

        // MidpointSlide::new should initialize borders based on the bounding box
        let ms = MidpointSlide::new(&mat, &units);

        assert_eq!(ms.borders.len(), 2);
        assert_eq!(ms.borders[0].min, 0.0);
        assert_eq!(ms.borders[0].max, 10.0);
    }

    #[test]
    fn test_midpoint_slide_split_behavior() {
        let mat = setup_test_matrix(); // (0,0) and (10,10)
        let mut units = vec![0, 1];
        let ms = MidpointSlide::new(&mat, &units);

        // Splitting should return a SplitUnit and two new MidpointSlide instances
        if let Some((split_unit, left_ms, right_ms)) = ms.split(&mat, &mut units) {
            let dim = split_unit.split.dimension;
            let val = split_unit.split.value;

            // For a 0.0 to 10.0 range, midpoint is 5.0
            assert_eq!(val, 5.0);

            // Check child border updates
            assert_eq!(left_ms.borders[dim].max, 5.0);
            assert_eq!(right_ms.borders[dim].min, 5.0);

            // Unit 0 (0.0) should be left, Unit 1 (10.0) should be right
            // Split index indicates the start of the right group
            assert_eq!(split_unit.unit, 1);
            assert_eq!(units[0], 0);
            assert_eq!(units[1], 1);
        } else {
            panic!("Should have found a split for distinct points");
        }
    }

    #[test]
    fn test_midpoint_slide_no_split_on_identical_points() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let mat = Matrix::new(data, nz(2)).unwrap();
        let mut units = vec![0, 1];

        let ms = MidpointSlide::new(&mat, &units);
        let result = ms.split(&mat, &mut units);

        // Cannot split if all points are at the same coordinate
        assert!(result.is_none());
    }
}
