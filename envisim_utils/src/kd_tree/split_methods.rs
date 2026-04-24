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

use super::PointAccess;

#[derive(Clone, Copy, Debug)]
pub struct Border {
    pub min: f64,
    pub max: f64,
}
impl Border {
    #[inline]
    pub fn range(&self) -> f64 { self.max - self.min }
    #[inline]
    pub fn centre(&self) -> f64 { (self.min + self.max) * 0.5 }
    /// Returns the centre if it is possible to place in (min, max), including some checks for
    /// seeing if max-min is large enough.
    #[inline]
    pub fn valid_centre(&self) -> Option<f64> {
        let centre = self.centre();
        let range = self.range();
        if !centre.is_finite() || !range.is_finite() || range <= f64::EPSILON {
            return None;
        }
        // Also reject when min/max are too close to distinguish
        let scale = self.min.abs().max(self.max.abs());
        // Consider larger epsilon?
        (scale * f64::EPSILON < range).then_some(centre)
    }
    #[inline]
    fn set_min(&mut self, v: f64) { self.min = self.min.min(v); }
    #[inline]
    fn set_max(&mut self, v: f64) { self.max = self.max.max(v); }
    fn from_data<T>(data: &T, units: &[usize], dim: usize) -> Self
    where
        T: PointAccess,
    {
        if units.is_empty() {
            return Self::default();
        }
        let mut b = Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        };
        for &id in units.iter() {
            b.set_min(data.coord(id, dim));
            b.set_max(data.coord(id, dim));
        }
        b
    }
    pub fn from_data_to_vec<T>(data: &T, units: &[usize]) -> Box<[Self]>
    where
        T: PointAccess,
    {
        (0..data.dim().get())
            .map(|d| Self::from_data(data, units, d))
            .collect()
    }
}
impl Default for Border {
    fn default() -> Self { Self { min: 0.0, max: 0.0 } }
}

#[derive(Clone, Debug, Copy)]
pub struct Split {
    dimension: usize,
    value: f64,
    /// If `true`, equal values goes to the left
    leq: bool,
}
impl Split {
    #[inline]
    pub fn dimension(&self) -> usize { self.dimension }
    #[inline]
    pub fn value(&self) -> f64 { self.value }
    #[inline]
    pub fn leq(&self) -> bool { self.leq }
    #[inline]
    pub fn value_distance(&self, value: f64) -> f64 { value - self.value }
    #[inline]
    pub fn unit_distance(&self, unit: &[f64]) -> f64 { self.value_distance(unit[self.dimension]) }
    #[inline]
    pub fn value_is_left(&self, value: f64) -> bool {
        let d = self.value_distance(value);
        d < 0.0 || (self.leq && d == 0.0)
    }
    #[inline]
    pub fn unit_is_left(&self, unit: &[f64]) -> bool { self.value_is_left(unit[self.dimension]) }
}
pub struct SplitUnit {
    split: Split,
    /// First unit to the right of the split. Must be in (0, len)
    unit: usize,
}
impl SplitUnit {
    #[inline]
    fn new(split: Split, unit: usize) -> Self { Self { split, unit } }
    #[inline]
    pub fn split(&self) -> &Split { &self.split }
    #[inline]
    pub fn dimension(&self) -> usize { self.split.dimension }
    #[inline]
    pub fn value(&self) -> f64 { self.split.value }
    #[inline]
    pub fn leq(&self) -> bool { self.split.leq }
    #[inline]
    pub fn unit(&self) -> usize { self.unit }
}
impl From<SplitUnit> for Split {
    fn from(su: SplitUnit) -> Self { su.split }
}

/// Type alias for split-finding function. `T` must implement [`PointAccess`].
/// `FindSplit` must return a valid SplitUnit, i.e 0 < SplitUnit.unit < len. Failure to do so is UB.
pub type FindSplit<T> = fn(&T, &[Border], &mut [usize]) -> Option<SplitUnit>;

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
pub fn midpoint_slide<T>(data: &T, borders: &[Border], units: &mut [usize]) -> Option<SplitUnit>
where
    T: PointAccess,
{
    assert_eq!(data.dim().get(), borders.len());

    if units.is_empty() {
        return None;
    }

    // Sort dims by range
    let border_indices: Box<[usize]> = {
        let mut indices: Box<[usize]> = (0..borders.len()).collect();
        indices.sort_unstable_by(|&a, &b| borders[b].range().total_cmp(&borders[a].range()));
        indices
    };

    let mut split = Split {
        dimension: 0,
        value: 0.0,
        leq: true,
    };

    for &dim_idx in border_indices.iter() {
        // If the current border is degenerate, we assume any subsequent borders to be degenerate as
        // well -- as indices is sorted by border range -- and do an early return.
        let centre = borders[dim_idx].valid_centre()?;

        split.dimension = dim_idx;
        split.value = centre;
        split.leq = true;

        // Returns the maximum value of the left units, and the minimum value of the right units,
        // partitions the units according to the split value, and sets split.unit to the
        // partitioning units
        let (mut split_unit, left_max, right_min): (usize, f64, f64) =
            midpoint_slide_sort(data, &split, units);

        // We have to degenerate cases:
        // When split_unit = 0, then all units are to the right.
        // When split_unit = len, then all units are to the left.
        // Neither of these degenerate cases should happen, as this implies that centre was off, and
        // the range is 0.0, but we take care of it anyway
        if split_unit == 0 {
            // Try setting the split to the min-value of the rights, and see if it's possible to
            // split the data there instead.
            split.value = right_min;
            (split_unit, _, _) = midpoint_slide_sort(data, &split, units);
            // If we failed, all units were moved to the left instead
            if split_unit == units.len() {
                continue;
            }
        } else if split_unit == units.len() {
            // Try setting the split to the max-value of the lefts, and see if it's possible to
            // split the data there instead.
            split.value = left_max;
            split.leq = false;
            (split_unit, _, _) = midpoint_slide_sort(data, &split, units);
            // If we failed, all units were moved to the right instead
            if split_unit == 0 {
                continue;
            }
        }

        // When we get here, it should be impossible for split_unit to be 0 or len
        return SplitUnit::new(split, split_unit).into();
    }

    None
}
/// Sorts the `units` in two ranges, such that all units with a value `< split.value` goes first.
/// Returns the tuple `(left_max, right_min)`, where
/// - `left_max` is the largest value in the `0..split_unit` set
/// - `right_min` is the smallest value in the `split_unit..` set
#[inline]
fn midpoint_slide_sort<T>(data: &T, split: &Split, units: &mut [usize]) -> (usize, f64, f64)
where
    T: PointAccess,
{
    let mut left: usize = 0;
    let mut right: usize = units.len();
    let mut left_max: f64 = f64::NEG_INFINITY;
    let mut right_min: f64 = f64::INFINITY;

    // Sort units so that we have
    // x < value is in range [0, l)
    // x > value is in range [r, n)
    // At end of loop: l == r
    while left < right {
        let v = data.coord(units[left], split.dimension);
        if v < split.value || (split.leq && v == split.value) {
            left_max = left_max.max(v);
            left += 1;
        } else {
            right_min = right_min.min(v);
            right -= 1;
            units.swap(left, right);
        }
    }

    (left, left_max, right_min)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::PointAccess;
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
        assert!(split.value_is_left(1.5));
        assert!(split.value_is_left(1.0));
        assert!(!split.value_is_left(2.0));
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
