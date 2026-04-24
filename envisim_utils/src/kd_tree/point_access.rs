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

pub trait PointAccess {
    /// Dimensions of point
    fn dim(&self) -> NonZeroUsize;
    /// Returns true of point id exists
    fn exists(&self, id: usize) -> bool;
    /// Returns the dimension value of point `id`
    #[inline]
    fn coord(&self, id: usize, dim: usize) -> f64 {
        self.try_coord(id, dim).expect("valid id and dim")
    }
    fn try_coord(&self, id: usize, dim: usize) -> Option<f64>;
    #[inline]
    fn sq_distance(&self, id: usize, point: &[f64]) -> f64 {
        assert_eq!(point.len(), self.dim().get());
        let mut sum = 0.0;
        for (d, &p) in point.iter().enumerate() {
            let diff = p - self.coord(id, d);
            sum += diff * diff;
        }
        sum
    }
    #[inline]
    fn try_sq_distance(&self, id: usize, point: &[f64]) -> Option<f64> {
        if !self.exists(id) || point.len() != self.dim().get() {
            return None;
        }
        self.sq_distance(id, point).into()
    }
    #[inline]
    fn to_boxed_slice(&self, id: usize) -> Option<Box<[f64]>> {
        self.exists(id)
            .then(|| (0..self.dim().get()).map(|d| self.coord(id, d)).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::PointAccess;
    use crate::matrix::Matrix;

    // Helper: build a simple 3x2 matrix (column-major):
    // Points (rows): [1,4], [2,5], [3,6]
    fn mat_3x2() -> Matrix<'static> {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        Matrix::from_vec(data, NonZeroUsize::new(3).unwrap()).unwrap()
    }

    #[test]
    fn dim_returns_column_count() {
        let m = mat_3x2();
        assert_eq!(m.dim().get(), 2);
    }

    #[test]
    fn exists_is_true_for_valid_ids() {
        let m = mat_3x2();
        assert!(m.exists(0));
        assert!(m.exists(1));
        assert!(m.exists(2));
    }

    #[test]
    fn exists_is_false_for_invalid_ids() {
        let m = mat_3x2();
        assert!(!m.exists(3));
        assert!(!m.exists(usize::MAX));
    }

    #[test]
    fn coord_returns_row_major_values_from_column_major_storage() {
        let m = mat_3x2();
        // Row 0 should be [1.0, 4.0]
        assert_eq!(m.coord(0, 0), 1.0);
        assert_eq!(m.coord(0, 1), 4.0);
        // Row 2 should be [3.0, 6.0]
        assert_eq!(m.coord(2, 0), 3.0);
        assert_eq!(m.coord(2, 1), 6.0);
    }

    #[test]
    fn try_coord_returns_some_for_valid_access() {
        let m = mat_3x2();
        assert_eq!(m.try_coord(1, 0), Some(2.0));
        assert_eq!(m.try_coord(1, 1), Some(5.0));
    }

    #[test]
    fn try_coord_returns_none_for_out_of_range_dim() {
        let m = mat_3x2();
        assert_eq!(m.try_coord(0, 2), None);
        assert_eq!(m.try_coord(0, 99), None);
    }

    #[test]
    fn sq_distance_is_zero_between_point_and_itself() {
        let m = mat_3x2();
        let p = [1.0, 4.0];
        assert_eq!(m.sq_distance(0, &p), 0.0);
    }

    #[test]
    fn sq_distance_computes_euclidean_squared() {
        let m = mat_3x2();
        // Row 0 is [1, 4]; query [4, 8] -> dx=3, dy=4 -> 9+16 = 25
        let p = [4.0, 8.0];
        assert_eq!(m.sq_distance(0, &p), 25.0);
    }

    #[test]
    #[should_panic]
    fn sq_distance_panics_on_dim_mismatch() {
        let m = mat_3x2();
        let p = [1.0]; // wrong length
        let _ = m.sq_distance(0, &p);
    }

    #[test]
    fn try_sq_distance_returns_none_for_bad_id_or_dim() {
        let m = mat_3x2();
        assert_eq!(m.try_sq_distance(99, &[0.0, 0.0]), None);
        assert_eq!(m.try_sq_distance(0, &[0.0]), None);
        assert_eq!(m.try_sq_distance(0, &[0.0, 0.0, 0.0]), None);
    }

    #[test]
    fn try_sq_distance_returns_some_for_valid_input() {
        let m = mat_3x2();
        let d = m.try_sq_distance(0, &[1.0, 4.0]);
        assert_eq!(d, Some(0.0));
    }

    #[test]
    fn to_boxed_slice_returns_full_row() {
        let m = mat_3x2();
        let s = m.to_boxed_slice(1).unwrap();
        assert_eq!(&*s, &[2.0, 5.0]);
    }

    #[test]
    fn to_boxed_slice_returns_none_for_invalid_id() {
        let m = mat_3x2();
        assert!(m.to_boxed_slice(99).is_none());
    }
}
