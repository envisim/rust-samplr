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

//! Utility functions for cube

use envisim_utils::indices::Indices;
use envisim_utils::matrix::{
    Dimensions,
    Matrix,
};

/// Set candidates from the indices list, i.e. a basic fallback
#[inline]
pub fn set_candidates_from_indices(candidates: &mut Vec<usize>, indices: &Indices, len: usize) {
    let number_of_remaining_units = indices.len();
    let len = if len == 0 || len > number_of_remaining_units {
        number_of_remaining_units
    } else {
        len
    };

    // Set candidates
    candidates.clear();
    candidates.extend_from_slice(&indices.list()[0..len]);
}
/// Finds a vector in null space of a (n-1)*n matrix. The matrix is mutated into rref.
///
/// # Panics
/// Panics if the matrix is not n-1 x n.
#[expect(
    clippy::float_cmp,
    reason = "reduced_row_echelon_form will have set some elements to 0/1 exactly"
)]
#[must_use]
#[inline]
pub fn find_vector_in_null_space(mat: &mut Matrix<f64>) -> Vec<f64> {
    let nrow = mat.nrow().get();
    let ncol = mat.ncol().get();
    assert!(nrow == ncol - 1, "the matrix must be n-1 x n");

    mat.reduced_row_echelon_form();
    let mut v = vec![1.0; ncol];

    // If (n-1, n-1) = 1.0, then we have linearly independent rows,
    // and the form of the matrix is an identity matrix with the parts
    // of the null space vector in the last column
    if mat[(nrow - 1, nrow - 1)] == 1.0 {
        for i in 0..nrow {
            v[i] = -mat[(i, ncol - 1)];
        }
        return v;
    }

    let mut pivot_cols = Vec::with_capacity(nrow);
    let mut is_pivot = vec![false; ncol];

    for row in 0..nrow {
        for col in 0..ncol {
            if mat[(row, col)] != 0.0 {
                // Found first non-zero entry in row
                if mat[(row, col)] == 1.0 {
                    pivot_cols.push(col);
                    is_pivot[col] = true;
                }
                break;
            }
        }
    }

    // Build null space vector
    // Free variables (non-pivot columns) alternating set to +/- 1
    // Basic vars (pivot cols) computet to satisfy Ax = 0

    // Set free variables
    let mut free_idx = true;
    for col in 0..ncol {
        if !is_pivot[col] {
            // set every other
            v[col] = if free_idx { 1.0 } else { -1.0 };
            free_idx = !free_idx;
        }
    }

    // Compute basic variables working backwards through rows
    for (row, &pivot_col) in pivot_cols.iter().enumerate().rev() {
        let mut sum = 0.0;
        for col in (pivot_col + 1)..ncol {
            sum += mat[(row, col)] * v[col];
        }
        v[pivot_col] = -sum;
    }

    v
}

#[cfg(test)]
mod tests {
    use envisim_utils::matrix::RawData;
    use envisim_utils::test_utils::*;

    use super::*;

    #[test]
    fn null() {
        let mut mat1 = Matrix::new(
            vec![
                1.0, 2.0, 3.0, 1.0, //
                5.0, 10.0, 1.0, 5.0, //
                10.0, 1.0, 5.0, 10.0, //
            ],
            3,
        )
        .unwrap();
        mat1.reduced_row_echelon_form();
        assert_vec!(
            mat1.data().data(),
            [
                1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
            ]
        );
        let mat1_nullvec = find_vector_in_null_space(&mut mat1);

        assert_mat!(
            mat1.mul_vec(&mat1_nullvec).unwrap(),
            Matrix::from_value(0.0, (nz(3), nz(1)))
        );

        let mut mat2 = Matrix::new(
            vec![
                1.0, 2.0, 3.0, 1.0, //
                5.0, 10.0, 10.0, 5.0, //
                1.0, 1.0, 5.0, 11.0, //
            ],
            3,
        )
        .unwrap();
        mat2.reduced_row_echelon_form();
        assert_mat!(
            mat2,
            Matrix::new(
                vec![
                    1., 0., 0., //
                    0., 1., 0., //
                    0., 0., 1., //
                    -2.5, 1.8333, 0.1666 //
                ],
                3
            )
            .unwrap(),
            1e-4
        );
        let mat2_nullvec = find_vector_in_null_space(&mut mat2);
        assert_mat!(
            mat2.mul_vec(&mat2_nullvec).unwrap(),
            Matrix::from_value(0.0, (nz(3), nz(1)))
        );
    }
}
