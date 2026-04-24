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

//! A matrix container

use std::borrow::Cow;
use std::num::NonZeroUsize;
use std::ops::{
    Index,
    IndexMut,
};

use crate::kd_tree::{
    PointAccess,
    Tree,
};

/// The shape (dimensions) of a matrix: rows × columns.
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
    fn from((rows, cols): (NonZeroUsize, NonZeroUsize)) -> Self { Self::new(rows, cols) }
}
impl From<MatrixDims> for (NonZeroUsize, NonZeroUsize) {
    fn from(value: MatrixDims) -> Self { (value.rows, value.cols) }
}
impl From<MatrixDims> for (usize, usize) {
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
    fn from((row, col): (usize, usize)) -> Self { Self::new(row, col) }
}
impl From<MatrixCoord> for (usize, usize) {
    fn from(value: MatrixCoord) -> Self { (value.row, value.col) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<'a> {
    data: Cow<'a, [f64]>,
    dims: MatrixDims,
}

impl<'a> Matrix<'a> {
    /// Clones the underlying data if it was borrowed
    #[inline]
    pub fn to_mut(&mut self) -> &mut Self {
        self.data.to_mut();
        self
    }
    /// Returns a reference to the matrix
    #[inline]
    pub fn clone_shallow(&self) -> Matrix<'_> {
        let data = match self.data {
            Cow::Borrowed(b) => Cow::Borrowed(b),
            Cow::Owned(ref b) => Cow::Borrowed(b.as_slice()),
        };

        Matrix {
            data,
            dims: self.dims,
        }
    }
    /// Constructs a new matrix, by borrowing the data.
    #[inline]
    pub fn new(data: &'a [f64], rows: NonZeroUsize) -> Option<Self> {
        let dims = MatrixDims::from_row_count(data.len(), rows)?;
        let m = Self {
            data: Cow::Borrowed(data),
            dims,
        };
        Some(m)
    }
    /// Constructs a new matrix, by moving the `data`.
    #[inline]
    pub fn from_vec(data: Vec<f64>, rows: NonZeroUsize) -> Option<Self> {
        let dims = MatrixDims::from_row_count(data.len(), rows)?;
        let m = Self {
            data: Cow::Owned(data),
            dims,
        };
        Some(m)
    }
    /// Constructs a new matrix of size `dims` filled with `data`
    #[inline]
    pub fn from_value(data: f64, dims: MatrixDims) -> Self {
        let size = dims.len().get();
        Self {
            data: Cow::Owned(vec![data; size]),
            dims,
        }
    }
    /// Returns the underlying data (stored in column major).
    #[inline]
    pub fn data(&self) -> &[f64] { self.data.as_ref() }
    /// Returns the underlying data (stored in column major).
    /// If the data is a borrowed, it is first cloned.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] { self.data.to_mut() }
    /// Returns the number of rows in the matrix
    #[inline]
    pub fn nrow(&self) -> NonZeroUsize { self.dims.rows }
    /// Returns the number of columns in the matrix
    #[inline]
    pub fn ncol(&self) -> NonZeroUsize { self.dims.cols }
    /// Returns the dimensions of the matrix
    #[inline]
    pub fn dims(&self) -> MatrixDims { self.dims }
    /// Returns an iterator on the row
    #[inline]
    pub fn row_iter(&'a self, row: usize) -> Option<RowIterator<'a>> {
        self.dims
            .contains_row(row)
            .then(|| RowIterator::new(self, row))
    }
    /// Returns an iterator on the column
    #[allow(clippy::iter_skip_zero)]
    #[inline]
    pub fn col_iter(&'a self, col: usize) -> Option<ColIterator<'a>> {
        self.dims
            .contains_col(col)
            .then(|| ColIterator::new(self, col))
    }
    /// Resizes the matrix, without guaranteeing the preservation of any data.
    /// If matrix is extend, data is cloned
    #[inline]
    pub fn resize(&mut self, dims: MatrixDims) {
        if dims == self.dims() {
            return;
        }

        let new_size = dims.len().get();
        self.data.to_mut().resize(new_size, 0.0);
        self.dims = dims;
    }
    /// Calculates the reduced row echelon form of the matrix, in place.
    pub fn reduced_row_echelon_form(&mut self) {
        let dims = self.dims();
        let index = |row, col| row + col * dims.rows.get();
        let data = self.data.to_mut();

        let mut lead: usize = 0;

        // We can skip som tolerance on equality checks, as we can guarantee that (some) are exactly
        // 0.0 or 1.0
        for row in 0..dims.rows.get() {
            if dims.cols.get() <= lead {
                return;
            }

            let mut i: usize = row;

            while data[index(i, lead)] == 0.0 {
                i += 1;

                if i == dims.rows.get() {
                    i = row;
                    lead += 1;
                }

                if lead == dims.cols.get() {
                    return;
                }
            }

            // Swap rows i and row
            if i != row {
                let mut index_i = i;
                let mut index_row = row;
                for _ in 0..dims.cols.get() {
                    data.swap(index_i, index_row);
                    index_i += dims.rows.get();
                    index_row += dims.rows.get();
                }
            }

            // Divide ROW by lead, assuming all is 0 before lead
            let mut index_row = index(row, lead);
            let lead_value = data[index_row];
            if lead_value != 1.0 {
                data[index_row] = 1.0;
                index_row += dims.rows.get();

                for _ in (lead + 1)..dims.cols.get() {
                    data[index_row] /= lead_value;
                    index_row += dims.rows.get();
                }
            }

            // Remove ROW from all other rows
            for j in 0..dims.rows.get() {
                if j == row {
                    continue;
                }

                let mut index_j = index(j, lead);

                let lead_multiplicator = data[index_j];
                if lead_multiplicator == 0.0 {
                    continue;
                }

                data[index_j] = 0.0;
                index_j += dims.rows.get();
                let mut index_row = index(row, lead + 1);

                for _ in (lead + 1)..dims.cols.get() {
                    data[index_j] -= data[index_row] * lead_multiplicator;
                    index_j += dims.rows.get();
                    index_row += dims.rows.get();
                }
            }

            lead += 1;
        }
    }
    /// Returns the squared euclidean distance between the `row` and the slice `unit`
    #[inline]
    pub fn distance_to_row(&self, row: usize, unit: &[f64]) -> Option<f64> {
        if !self.dims.contains_row(row) || unit.len() != self.ncol().get() {
            return None;
        }

        let mut idx = row;
        let mut sum = 0.0;
        for &p in unit.iter() {
            let diff = p - self.data[idx];
            sum += diff * diff;
            idx += self.dims.rows.get();
        }
        Some(sum)
    }
    /// Returns the squared euclidean distance between the `row` and the slice `unit`
    #[inline]
    pub fn distance_between_rows(&self, row_a: usize, row_b: usize) -> Option<f64> {
        if !self.dims.contains_row(row_a) || !self.dims.contains_row(row_b) {
            return None;
        } else if row_a == row_b {
            return Some(0.0);
        }

        let mut idx = row_a.min(row_b);
        // One unit will always be offset by idx_diff cmp. idx
        let idx_diff = row_a.abs_diff(row_b);
        let mut sum = 0.0;
        for _ in 0..self.dims.cols.get() {
            let diff = self.data[idx] - self.data[idx + idx_diff];
            sum += diff * diff;
            idx += self.dims.rows.get();
        }
        Some(sum)
    }
    /// Performs the calculation of self * multiplicand, where self is a matrix A, and multiplicand
    /// is a vector.
    #[inline]
    pub fn prod_vec(&self, multiplicand: &[f64]) -> Option<Vec<f64>> {
        if multiplicand.len() != self.ncol().get() {
            return None;
        }

        let data = self.data();
        let mut prod = vec![0.0; self.nrow().get()];
        let mut index: usize = 0;

        for &mul in multiplicand.iter() {
            for pr in prod.iter_mut() {
                *pr += mul * data[index];
                index += 1;
            }
        }

        Some(prod)
    }
    /// Performes the calculation of matrices self * mat
    #[inline]
    pub fn mult(&'a self, mat: &Matrix) -> Option<Matrix<'static>> {
        if self.ncol() != mat.nrow() {
            return None;
        }

        let mut prod = Vec::<f64>::with_capacity(self.nrow().get() * mat.ncol().get());

        let mut index = 0usize;
        for _ in 0..mat.ncol().get() {
            prod.extend_from_slice(&self.prod_vec(&mat.data()[index..(index + mat.nrow().get())])?);
            index += mat.nrow().get();
        }

        Matrix::from_vec(prod, self.nrow())
    }
}

impl<'a> Index<MatrixCoord> for Matrix<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, idx: MatrixCoord) -> &Self::Output {
        let index = idx.to_linear(self.dims).unwrap();
        &self.data()[index]
    }
}
impl<'a> IndexMut<MatrixCoord> for Matrix<'a> {
    #[inline]
    fn index_mut(&mut self, idx: MatrixCoord) -> &mut Self::Output {
        let index = idx.to_linear(self.dims).unwrap();
        &mut self.data_mut()[index]
    }
}

pub struct RowIterator<'a> {
    data: &'a Matrix<'a>,
    pos: usize,
}
impl<'a> RowIterator<'a> {
    #[inline]
    pub fn new(matrix: &'a Matrix, row: usize) -> Self {
        Self {
            data: matrix,
            pos: row,
        }
    }
}
impl<'a> Iterator for RowIterator<'a> {
    type Item = &'a f64;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.data.data.get(self.pos)?;
        self.pos += self.data.nrow().get();
        Some(val)
    }
}
impl<'a> ExactSizeIterator for RowIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        let diff = self.data.data.len() - self.pos;
        let d = diff / self.data.nrow();
        let r = diff % self.data.nrow();
        if r > 0 { d + 1 } else { d }
    }
}
pub struct ColIterator<'a> {
    data: &'a [f64],
    pos: usize,
}
impl<'a> ColIterator<'a> {
    #[inline]
    pub fn new(matrix: &'a Matrix, col: usize) -> Self {
        let start = col * matrix.nrow().get();
        Self {
            data: &matrix.data[start..(start + matrix.nrow().get())],
            pos: 0,
        }
    }
}
impl<'a> Iterator for ColIterator<'a> {
    type Item = &'a f64;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.data.get(self.pos)?;
        self.pos += 1;
        Some(val)
    }
}
impl<'a> ExactSizeIterator for ColIterator<'a> {
    #[inline]
    fn len(&self) -> usize { self.data.len() - self.pos }
}

impl<'a> PointAccess for Matrix<'a> {
    #[inline]
    fn dim(&self) -> NonZeroUsize { self.dims.cols }
    #[inline]
    fn exists(&self, id: usize) -> bool { id < self.dims.rows.get() }
    #[inline]
    fn coord(&self, id: usize, dim: usize) -> f64 {
        let idx = id + dim * self.dims.rows.get();
        self.data[idx]
    }
    #[inline]
    fn try_coord(&self, id: usize, dim: usize) -> Option<f64> {
        let idx = id + dim * self.dims.rows.get();
        self.data.get(idx).copied()
    }
}

pub type MatrixTree<'a> = Tree<'a, Matrix<'a>>;
