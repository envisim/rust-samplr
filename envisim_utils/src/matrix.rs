// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
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

//! Two different matrix containers are provided:
//! - [`Matrix`], which is a mutable matrix owning it's own storage.
//! - [`RefMatrix`], which provides matrix operations on a provided, immutable vector.

use std::borrow::Cow;
use std::iter::{
    Skip,
    StepBy,
};
use std::ops::{
    Index,
    IndexMut,
};
use std::slice::Iter;

/// Matrix dimensions `(row, col)`
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixIndex(pub usize, pub usize);

impl MatrixIndex {
    #[inline]
    pub fn try_from_slice(v: &[f64], rows: usize) -> Option<Self> {
        if rows == 0 || v.len() % rows > 0 {
            return None;
        }
        let cols = v.len() / rows;

        Some(MatrixIndex(rows, cols))
    }
    #[inline]
    pub fn row(&self) -> usize { self.0 }
    #[inline]
    pub fn col(&self) -> usize { self.1 }
    #[inline]
    pub fn size(&self) -> usize { self.row() * self.col() }
    #[inline]
    pub fn transpose(&self) -> Self { MatrixIndex(self.1, self.0) }
    #[inline]
    pub fn to_index(&self, size: impl Into<MatrixIndex>) -> Option<usize> {
        let size = size.into();
        (self.row() < size.row() && self.col() < size.col())
            .then_some(self.row() + self.col() * size.row())
    }
    #[inline]
    pub fn new(idx: impl Into<MatrixIndex>) -> Self { idx.into() }
    #[inline]
    pub fn into_index(idx: impl Into<MatrixIndex>, size: impl Into<MatrixIndex>) -> Option<usize> {
        let idx = idx.into();
        let size = size.into();
        (idx.row() < size.row() && idx.col() < size.col())
            .then_some(idx.row() + idx.col() * size.row())
    }
}

impl From<(usize, usize)> for MatrixIndex {
    fn from(idx: (usize, usize)) -> Self { MatrixIndex(idx.0, idx.1) }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<'a> {
    data: Cow<'a, [f64]>,
    dims: MatrixIndex,
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
    pub fn new(data: &'a [f64], rows: usize) -> Option<Self> {
        let dims = MatrixIndex::try_from_slice(data, rows)?;
        let m = Self {
            data: Cow::Borrowed(data),
            dims,
        };
        Some(m)
    }
    /// Constructs a new matrix, by moving the `data`.
    #[inline]
    pub fn from_vec(data: Vec<f64>, rows: usize) -> Option<Self> {
        let dims = MatrixIndex::try_from_slice(&data, rows)?;
        let m = Self {
            data: Cow::Owned(data),
            dims,
        };
        Some(m)
    }
    /// Constructs a new matrix of size `dims` filled with `data`
    #[inline]
    pub fn from_value(data: f64, dims: impl Into<MatrixIndex>) -> Option<Self> {
        let dims = dims.into();
        let size = dims.size();
        if size == 0 {
            return None;
        }

        let m = Self {
            data: Cow::Owned(vec![data; size]),
            dims,
        };
        Some(m)
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
    pub fn nrow(&self) -> usize { self.dims.row() }
    /// Returns the number of columns in the matrix
    #[inline]
    pub fn ncol(&self) -> usize { self.dims.col() }
    /// Returns the dimensions of the matrix
    #[inline]
    pub fn dims(&self) -> MatrixIndex { self.dims }
    /// Returns an iterator on the row
    #[inline]
    pub fn row_iter(&'a self, row: usize) -> MatrixIterator<'a> {
        assert!(row < self.nrow());
        MatrixIterator {
            iter: self.data().iter().skip(row).step_by(self.nrow()),
            dims: self.dims(),
            step: self.nrow(),
            count: 0,
        }
    }
    /// Returns an iterator on the column
    #[allow(clippy::iter_skip_zero)]
    #[inline]
    pub fn col_iter(&'a self, col: usize) -> MatrixIterator<'a> {
        assert!(col < self.ncol());
        MatrixIterator {
            iter: self.data()[(self.nrow() * col)..(self.nrow() * (col + 1))]
                .iter()
                .skip(0)
                .step_by(1),
            dims: self.dims(),
            step: 1,
            count: 0,
        }
    }
    /// Resizes the matrix, without guaranteeing the preservation of any data.
    /// If matrix is extend, data is cloned
    #[inline]
    pub fn resize(&mut self, dims: impl Into<MatrixIndex>) -> Option<&mut Self> {
        let dims = dims.into();
        if dims.row() == 0 || dims.col() == 0 {
            return None;
        } else if dims == self.dims() {
            return Some(self);
        }

        let new_size = dims.size();

        self.data.to_mut().resize(new_size, 0.0);
        self.dims = dims;
        Some(self)
    }
    /// Calculates the reduced row echelon form of the matrix, in place.
    pub fn reduced_row_echelon_form(&mut self) {
        let dims = self.dims();
        let index = |row, col| row + col * dims.row();
        let data = self.data.to_mut();

        let mut lead: usize = 0;

        for row in 0..dims.row() {
            if dims.col() <= lead {
                return;
            }

            let mut i: usize = row;

            while data[index(i, lead)] == 0.0 {
                i += 1;

                if i == dims.row() {
                    i = row;
                    lead += 1;
                }

                if lead == dims.col() {
                    return;
                }
            }

            // Swap rows i and row
            if i != row {
                let mut index_i = i;
                let mut index_row = row;
                for _ in 0..dims.col() {
                    data.swap(index_i, index_row);
                    index_i += dims.row();
                    index_row += dims.row();
                }
            }

            // Divide ROW by lead, assuming all is 0 before lead
            let mut index_row = index(row, lead);
            let lead_value = unsafe { *data.get_unchecked(index_row) };
            if lead_value != 1.0 {
                data[index_row] = 1.0;
                index_row += dims.row();

                for _ in (lead + 1)..dims.col() {
                    data[index_row] /= lead_value;
                    index_row += dims.row();
                }
            }

            // Remove ROW from all other rows
            for j in 0..dims.row() {
                if j == row {
                    continue;
                }

                let mut index_j = index(j, lead);

                let lead_multiplicator = data[index_j];
                if lead_multiplicator == 0.0 {
                    continue;
                }

                data[index_j] = 0.0;
                index_j += dims.row();
                let mut index_row = index(row, lead + 1);

                for _ in (lead + 1)..dims.col() {
                    data[index_j] -= data[index_row] * lead_multiplicator;
                    index_j += dims.row();
                    index_row += dims.row();
                }
            }

            lead += 1;
        }
    }
    /// Returns the squared eculidean distance between the `row` and the slice `unit`
    #[inline]
    pub fn distance_to_row(&self, row: usize, unit: &[f64]) -> Option<f64> {
        if unit.len() != self.ncol() || row >= self.nrow() {
            return None;
        }

        let v = self
            .row_iter(row)
            .zip(unit.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2));
        Some(v)
    }
    /// Returns the squared eculidean distance between the `row` and the slice `unit`
    #[inline]
    pub fn distance_between_rows(&self, row_a: usize, row_b: usize) -> Option<f64> {
        if row_a >= self.nrow() || row_b >= self.nrow() {
            return None;
        } else if row_a == row_b {
            return Some(0.0);
        }

        let v = self
            .row_iter(row_a)
            .zip(self.row_iter(row_b))
            .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2));
        Some(v)
    }
    /// Performes the calculation of self * multiplicand, where self is a matrix A, and multiplicand
    /// is a vector.
    #[inline]
    pub fn prod_vec(&self, multiplicand: &[f64]) -> Option<Vec<f64>> {
        if multiplicand.len() != self.ncol() {
            return None;
        }

        let data = self.data();
        let mut prod = vec![0.0; self.nrow()];
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
    pub fn mult<'b>(&'a self, mat: &Matrix) -> Option<Matrix<'b>> {
        if self.ncol() != mat.nrow() {
            return None;
        }

        let mut prod = Vec::<f64>::with_capacity(self.nrow() * mat.ncol());

        let mut index = 0usize;
        for _ in 0..mat.ncol() {
            prod.extend_from_slice(&self.prod_vec(&mat.data()[index..(index + mat.nrow())])?);
            index += mat.nrow();
        }

        Matrix::from_vec(prod, self.nrow())
    }
}

impl<'a> Index<MatrixIndex> for Matrix<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, idx: MatrixIndex) -> &f64 {
        let index = idx.to_index(self.dims()).unwrap();
        &self.data()[index]
    }
}
impl<'a> IndexMut<MatrixIndex> for Matrix<'a> {
    #[inline]
    fn index_mut(&mut self, idx: MatrixIndex) -> &mut f64 {
        let index = idx.to_index(self.dims()).unwrap();
        &mut self.data_mut()[index]
    }
}

impl<'a> Index<(usize, usize)> for Matrix<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, idx: (usize, usize)) -> &f64 {
        let index = MatrixIndex::into_index(idx, self.dims()).unwrap();
        &self.data()[index]
    }
}
impl<'a> IndexMut<(usize, usize)> for Matrix<'a> {
    #[inline]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut f64 {
        let index = MatrixIndex::into_index(idx, self.dims()).unwrap();
        &mut self.data_mut()[index]
    }
}

pub struct MatrixIterator<'a> {
    iter: StepBy<Skip<Iter<'a, f64>>>,
    dims: MatrixIndex,
    step: usize,
    count: usize,
}
impl<'a> MatrixIterator<'a> {
    #[inline]
    pub fn dims(&self) -> MatrixIndex {
        if self.step == 1 {
            MatrixIndex(self.dims.row(), 1)
        } else {
            MatrixIndex(1, self.dims.col())
        }
    }
}
impl<'a> Iterator for MatrixIterator<'a> {
    type Item = &'a f64;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.iter.next()
    }
}
impl<'a> ExactSizeIterator for MatrixIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        if self.step == 1 {
            self.dims.row() - self.count
        } else {
            self.dims.col() - self.count
        }
    }
}
