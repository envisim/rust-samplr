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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MatrixDims {
    pub rows: NonZeroUsize,
    pub cols: NonZeroUsize,
}
impl MatrixDims {
    #[inline]
    pub fn new(rows: NonZeroUsize, cols: NonZeroUsize) -> Self { Self { rows, cols } }
    #[inline]
    pub fn try_new(rows: usize, cols: usize) -> Option<Self> {
        let rows = NonZeroUsize::new(rows)?;
        let cols = NonZeroUsize::new(cols)?;
        Self::new(rows, cols).into()
    }
    /// Infer shape from a slice length and row count. Returns `None` if rows doesn't divide `len`
    /// evenly.
    #[inline]
    pub fn from_row_count(len: usize, rows: NonZeroUsize) -> Option<Self> {
        if len % rows != 0 {
            return None;
        }
        let len = NonZeroUsize::new(len)?;
        let cols = NonZeroUsize::new(len.get() / rows)?;
        Self::new(rows, cols).into()
    }
    /// Total number of elements
    #[inline]
    pub fn len(&self) -> NonZeroUsize { self.rows.saturating_mul(self.cols) }
    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
        }
    }
    #[inline]
    pub fn contains_row(&self, row: usize) -> bool { row < self.rows.get() }
    #[inline]
    pub fn contains_col(&self, col: usize) -> bool { col < self.cols.get() }
    /// Returns `true` if `coord` falls inside the shape
    #[inline]
    pub fn contains(&self, coord: MatrixCoord) -> bool {
        self.contains_row(coord.row) && self.contains_col(coord.col)
    }
}
impl From<(NonZeroUsize, NonZeroUsize)> for MatrixDims {
    #[inline]
    fn from((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self { Self::new(rows, cols) }
}
impl From<MatrixDims> for (NonZeroUsize, NonZeroUsize) {
    #[inline]
    fn from(value: MatrixDims) -> Self { (value.rows, value.cols) }
}
impl From<MatrixDims> for (usize, usize) {
    #[inline]
    fn from(value: MatrixDims) -> Self { (value.rows.get(), value.cols.get()) }
}

/// A position within a matrix: (row, col), zero-indexed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MatrixCoord {
    pub row: usize,
    pub col: usize,
}
impl MatrixCoord {
    #[inline]
    pub fn new(row: usize, col: usize) -> Self { Self { row, col } }
    /// Convert to a linear index in column-major order. Returns `None` if the coordinate is out of
    /// bounds.
    #[inline]
    pub fn to_linear(&self, shape: MatrixDims) -> Option<usize> {
        shape
            .contains(*self)
            .then(|| self.row + self.col * shape.rows.get())
    }

    /// Convert from a linear index in column-major order. Returns `None` if `index >= shape.len()`.
    #[inline]
    pub fn from_linear(index: usize, shape: MatrixDims) -> Option<Self> {
        // Rem<NonZeroUsize> is in rust since 1.51
        (index < shape.len().get()).then(|| Self::new(index % shape.rows, index / shape.rows))
    }
}
impl From<(usize, usize)> for MatrixCoord {
    #[inline]
    fn from((row, col): (usize, usize)) -> Self { Self::new(row, col) }
}
impl From<MatrixCoord> for (usize, usize) {
    #[inline]
    fn from(value: MatrixCoord) -> Self { (value.row, value.col) }
}
