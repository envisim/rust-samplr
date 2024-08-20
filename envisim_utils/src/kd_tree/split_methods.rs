// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

use crate::matrix::{OperateMatrix, RefMatrix};

// (unit, variable, value)
#[derive(Clone, Debug)]
pub struct Split {
    pub unit: usize,
    pub dimension: usize,
    pub value: f64,
    pub leq: bool,
}
pub type FindSplit = fn(&[(f64, f64)], &RefMatrix, &mut [usize]) -> Option<Split>;

/// The midpoint slide splitting method.
/// Returns a split, where units `[0..unit)` have values < `value`, and units [unit,..) have values
/// > `value`.
/// If `leq` is `true`, the first group also contains equal elements, otherwise the right group
/// contains equal elements.
///
/// Returns `None` if no such split exists
///
/// # References
/// Maneewongvatana, S., & Mount, D. M. (1999).
/// It’s okay to be skinny, if your friends are fat.
/// In Center for geometric computing 4th annual workshop on computational geometry (Vol. 2).
pub fn midpoint_slide(
    borders: &[(f64, f64)],
    data: &RefMatrix,
    units: &mut [usize],
) -> Option<Split> {
    assert_eq!(data.ncol(), borders.len());

    if units.is_empty() {
        return None;
    }

    let mut sorted_dims: Vec<(usize, f64)> =
        borders.iter().map(|&b| b.1 - b.0).enumerate().collect();
    sorted_dims.sort_unstable_by(|a, b| (b.1).partial_cmp(&a.1).unwrap());

    let mut split = Split {
        unit: 0,
        dimension: 0,
        value: 0.0,
        leq: true,
    };

    for dims in sorted_dims.iter() {
        let spread = dims.1;

        if spread <= f64::EPSILON {
            return None;
        }

        split.dimension = dims.0;
        split.value = spread * 0.5 + borders[split.dimension].0;
        split.leq = true;

        let left_max: f64;
        let right_min: f64;

        (left_max, right_min) = midpoint_slide_sort(data, units, &mut split);

        if split.unit == 0 {
            split.value = right_min;
            midpoint_slide_sort(data, units, &mut split);

            if split.unit == units.len() {
                continue;
            }
        } else if split.unit == units.len() {
            split.value = left_max;
            split.leq = false;
            midpoint_slide_sort(data, units, &mut split);

            if split.unit == 0 {
                continue;
            }
        }

        return Some(split);
    }

    None
}
/// Sorts the `units` in two ranges, such that all units with a value `< split.value` goes first.
/// Returns the tuple `(left_max, right_min)`, where
/// - `left_max` is the largest value in the `0..split_unit` set
/// - `right_min` is the smallest value in the `split_unit..` set
#[inline]
fn midpoint_slide_sort(data: &RefMatrix, units: &mut [usize], split: &mut Split) -> (f64, f64) {
    split.unit = 0;
    let mut right: usize = units.len();
    let mut left_max: f64 = f64::MIN;
    let mut right_min: f64 = f64::MAX;

    // Sort units so that we have
    // x < value is in range [0, l)
    // x > value is in range [r, n)
    // At end of loop: right == split.unit
    while split.unit < right {
        let v = data[(units[split.unit], split.dimension)];
        if v < split.value || (split.leq && v == split.value) {
            if v > left_max {
                left_max = v;
            }

            split.unit += 1;
        } else {
            if v < right_min {
                right_min = v;
            }

            right -= 1;
            units.swap(split.unit, right);
        }
    }

    (left_max, right_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_slide() {
        let v = vec![0.0, 1.0, 2.0, 13.0];
        let m = RefMatrix::new(&v, 4);
        let split = super::midpoint_slide(&vec![(0.0, 13.0)], &m, &mut vec![0, 1, 2, 3]).unwrap();
        assert_eq!(split.unit, 3);
        assert_eq!(split.dimension, 0);
        assert_eq!(split.value, 6.5);

        let v = vec![0.0, 1.0, 2.0, 13.0, 0.0, 10.0, 20.0, 30.0];
        let m = RefMatrix::new(&v, 4);
        let split =
            super::midpoint_slide(&vec![(0.0, 13.0), (0.0, 30.0)], &m, &mut vec![0, 1, 2, 3])
                .unwrap();
        assert_eq!(split.unit, 2);
        assert_eq!(split.dimension, 1);
        assert_eq!(split.value, 15.0);

        let v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let m = RefMatrix::new(&v, 3);
        let split = super::midpoint_slide(&vec![(0.0, 0.0), (1.0, 1.0)], &m, &mut vec![0, 1, 2]);
        assert!(split.is_none());
    }
}
