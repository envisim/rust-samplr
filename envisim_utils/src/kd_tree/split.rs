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

//! Defines a split, and a split with a unit

use crate::utils::{
    Number,
    PointSet,
};

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
    pub(super) fn set_unit<P>(&mut self, data: &P, units: &mut [P::Id])
    where
        N: Number,
        P: PointSet<Value = N>,
    {
        let mut left: usize = 0;
        let mut right: usize = units.len();
        while left < right {
            // SAFETY:
            // units have been checked on construction of tree
            let v = unsafe { data.coord_unchecked(units[left], self.split.dimension) };
            if self.split.is_left(*v) {
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
