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

//! Matrix dimension and indexing

use std::num::NonZeroUsize;

/// Dimensions of a matrix
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MatrixDims {
    /// The number of rows
    pub rows: NonZeroUsize,
    /// The number of columns
    pub cols: NonZeroUsize,
}
impl MatrixDims {
    /// Constructs a new dimension
    #[inline]
    pub fn new(rows: NonZeroUsize, cols: NonZeroUsize) -> Self { Self { rows, cols } }
    /// Constructs a new dimension if the rows/cols are non-zero
    #[must_use]
    #[inline]
    pub fn try_new(rows: usize, cols: usize) -> Option<Self> {
        let rows = NonZeroUsize::new(rows)?;
        let cols = NonZeroUsize::new(cols)?;
        Self::new(rows, cols).into()
    }
    /// Infer dimensions from some data length and row count.
    /// Returns `None` if rows doesn't divide `len` evenly.
    #[must_use]
    #[inline]
    pub fn from_row_count(len: usize, rows: NonZeroUsize) -> Option<Self> {
        if len % rows != 0 {
            return None;
        }
        let len = NonZeroUsize::new(len)?;
        let cols = NonZeroUsize::new(len.get() / rows)?;
        Self::new(rows, cols).into()
    }
    /// Returns the total number of elements
    #[must_use]
    #[inline]
    pub fn len(&self) -> NonZeroUsize { self.rows.saturating_mul(self.cols) }
    /// Transposes the dimension
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
        }
    }
    /// Returns `true` if `row` would be contained within the dimensions.
    #[must_use]
    #[inline]
    pub fn contains_row(&self, row: usize) -> bool { row < self.rows.get() }
    /// Returns `true` if `col` would be contained within the dimensions.
    #[must_use]
    #[inline]
    pub fn contains_col(&self, col: usize) -> bool { col < self.cols.get() }
    /// Returns `true` if `coord` would be contained within the dimensions.
    #[must_use]
    #[inline]
    pub fn contains(&self, coord: MatrixCoord) -> bool {
        self.contains_row(coord.row) && self.contains_col(coord.col)
    }
}
impl From<(NonZeroUsize, NonZeroUsize)> for MatrixDims {
    /// Converts from a non-zero pair into [`MatrixDims`]
    #[inline]
    fn from((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self { Self::new(rows, cols) }
}
impl From<MatrixDims> for (NonZeroUsize, NonZeroUsize) {
    /// Converts from [`MatrixDims`] into a non-zero pair
    #[inline]
    fn from(value: MatrixDims) -> Self { (value.rows, value.cols) }
}
impl From<MatrixDims> for (usize, usize) {
    /// Converts from [`MatrixDims`] into a pair
    #[inline]
    fn from(value: MatrixDims) -> Self { (value.rows.get(), value.cols.get()) }
}

pub trait Dimensions {
    /// Returns the dimensions of the object
    fn dims(&self) -> MatrixDims;
    /// Returns the number of rows of the object
    #[must_use]
    #[inline]
    fn nrow(&self) -> NonZeroUsize { self.dims().rows }
    /// Returns the number of columns of the object
    #[must_use]
    #[inline]
    fn ncol(&self) -> NonZeroUsize { self.dims().cols }
}
impl<D> Dimensions for &D
where
    D: Dimensions + ?Sized,
{
    #[inline]
    fn dims(&self) -> MatrixDims { (**self).dims() }
    #[inline]
    fn nrow(&self) -> NonZeroUsize { (**self).nrow() }
    #[inline]
    fn ncol(&self) -> NonZeroUsize { (**self).ncol() }
}

/// A zero-indexed position within a matrix: (`row`, `col`)
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MatrixCoord {
    /// Row index
    pub row: usize,
    /// Column index
    pub col: usize,
}
impl MatrixCoord {
    /// Constructs a new coordinate from `row` and `col`
    #[inline]
    pub fn new(row: usize, col: usize) -> Self { Self { row, col } }
    /// Convert to a linear index in column-major order.
    /// Returns `None` if the coordinate is out of bounds with respect to the shape.
    #[must_use]
    #[inline]
    pub fn to_linear(&self, shape: MatrixDims) -> Option<usize> {
        shape
            .contains(*self)
            .then(|| self.row + self.col * shape.rows.get())
    }

    /// Convert from a linear index in column-major order.
    /// Returns `None` if `index` cannot be contained in `shape`.
    #[must_use]
    #[inline]
    pub fn from_linear(index: usize, shape: MatrixDims) -> Option<Self> {
        // Rem<NonZeroUsize> is in rust since 1.51
        (index < shape.len().get()).then(|| Self::new(index % shape.rows, index / shape.rows))
    }
}
impl From<(usize, usize)> for MatrixCoord {
    /// Converts from a row-col-pair into a coordinate
    #[inline]
    fn from((row, col): (usize, usize)) -> Self { Self::new(row, col) }
}
impl From<MatrixCoord> for (usize, usize) {
    /// Converts from a coordinate into a row-col-pair
    #[inline]
    fn from(value: MatrixCoord) -> Self { (value.row, value.col) }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    // Helper to create NonZeroUsize
    fn nz(n: usize) -> NonZeroUsize { NonZeroUsize::new(n).unwrap() }

    #[test]
    fn test_matrix_dims_new() {
        let dims = MatrixDims::new(nz(2), nz(3));
        assert_eq!(dims.rows.get(), 2);
        assert_eq!(dims.cols.get(), 3);
    }

    #[test]
    fn test_matrix_dims_try_new() {
        assert!(MatrixDims::try_new(2, 3).is_some());
        assert!(MatrixDims::try_new(0, 3).is_none());
        assert!(MatrixDims::try_new(2, 0).is_none());
    }

    #[test]
    fn test_matrix_dims_from_row_count() {
        let dims = MatrixDims::from_row_count(6, nz(2)).unwrap();
        assert_eq!(dims.cols.get(), 3);

        // Fails if not evenly divisible
        assert!(MatrixDims::from_row_count(5, nz(2)).is_none());
        // Fails if resulting columns would be zero (len is 0)
        assert!(MatrixDims::from_row_count(0, nz(2)).is_none());
    }

    #[test]
    fn test_matrix_dims_len() {
        let dims = MatrixDims::new(nz(2), nz(4));
        assert_eq!(dims.len().get(), 8);
    }

    #[test]
    fn test_matrix_dims_transpose() {
        let dims = MatrixDims::new(nz(2), nz(3));
        let transposed = dims.transpose();
        assert_eq!(transposed.rows.get(), 3);
        assert_eq!(transposed.cols.get(), 2);
    }

    #[test]
    fn test_matrix_dims_bounds_checks() {
        let dims = MatrixDims::new(nz(2), nz(3));

        assert!(dims.contains_row(0));
        assert!(dims.contains_row(1));
        assert!(!dims.contains_row(2));

        assert!(dims.contains_col(0));
        assert!(dims.contains_col(2));
        assert!(!dims.contains_col(3));

        assert!(dims.contains(MatrixCoord::new(1, 2)));
        assert!(!dims.contains(MatrixCoord::new(2, 2)));
    }

    #[test]
    fn test_matrix_coord_to_linear() {
        let shape = MatrixDims::new(nz(3), nz(2)); // 3 rows, 2 cols
        // Column-major: (r0, c0), (r1, c0), (r2, c0), (r0, c1), (r1, c1), (r2, c1)

        assert_eq!(MatrixCoord::new(0, 0).to_linear(shape), Some(0));
        assert_eq!(MatrixCoord::new(1, 0).to_linear(shape), Some(1));
        assert_eq!(MatrixCoord::new(0, 1).to_linear(shape), Some(3));
        assert_eq!(MatrixCoord::new(2, 1).to_linear(shape), Some(5));

        // Out of bounds
        assert_eq!(MatrixCoord::new(3, 0).to_linear(shape), None);
        assert_eq!(MatrixCoord::new(0, 2).to_linear(shape), None);
    }

    #[test]
    fn test_matrix_coord_from_linear() {
        let shape = MatrixDims::new(nz(3), nz(2));

        assert_eq!(
            MatrixCoord::from_linear(0, shape),
            Some(MatrixCoord::new(0, 0))
        );
        assert_eq!(
            MatrixCoord::from_linear(1, shape),
            Some(MatrixCoord::new(1, 0))
        );
        assert_eq!(
            MatrixCoord::from_linear(3, shape),
            Some(MatrixCoord::new(0, 1))
        );
        assert_eq!(
            MatrixCoord::from_linear(5, shape),
            Some(MatrixCoord::new(2, 1))
        );

        // Out of bounds (len is 6, so index 6 is invalid)
        assert_eq!(MatrixCoord::from_linear(6, shape), None);
    }
}
