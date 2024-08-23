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

//! Two different matrix containers are provided:
//! - [`Matrix`], which is a mutable matrix owning it's own storage.
//! - [`RefMatrix`], which provides matrix operations on a provided, immutable vector.

use std::iter::{Skip, StepBy};
use std::ops::{Index, IndexMut};
use std::slice::Iter;

/// Matrix dimensions `(row, col)`
type MatrixIndex = (usize, usize);

#[allow(clippy::exhaustive_enums)]
pub enum MatrixData<'a> {
    Mutable(Vec<f64>),
    Reference(&'a [f64]),
}

pub struct Matrix<'a> {
    data: MatrixData<'a>,
    rows: usize,
    cols: usize,
}

impl<'a> Matrix<'a> {
    #[inline]
    fn matrix_index(&self, (row, col): MatrixIndex) -> usize {
        assert!(col < self.cols, "col {} larger than max {}", col, self.cols);
        assert!(row < self.rows, "row {} larger than max {}", row, self.rows);
        row + col * self.rows
    }
    // #[inline]
    // unsafe fn matrix_index_unchecked(&self, (row, col): MatrixIndex) -> usize {
    //     col * self.rows + row
    // }
    /// If the matrix is a reference matrix, this method clones the underlying data and transforms
    /// self into a mutable matrix.
    /// Calling the method on a mutable matrix is a noop.
    #[inline]
    pub fn to_mut(&mut self) -> &Self {
        if let MatrixData::Reference(v) = self.data {
            self.data = MatrixData::Mutable(v.to_vec());
        }
        self
    }
    /// Constructs a new mutable matrix, by copying the `data`.
    #[inline]
    pub fn new(data: &[f64], rows: usize) -> Self {
        assert!(rows > 0);
        assert!(data.len() % rows == 0);
        let cols = data.len() / rows;
        Self {
            data: MatrixData::Mutable(data.to_vec()),
            rows,
            cols,
        }
    }
    /// Constructs a new mutable matrix, by moving the `data`.
    #[inline]
    pub fn from_vec(data: Vec<f64>, rows: usize) -> Self {
        assert!(rows > 0);
        assert!(data.len() % rows == 0);
        let cols = data.len() / rows;
        Self {
            data: MatrixData::Mutable(data),
            rows,
            cols,
        }
    }
    /// Constructs a new mutable matrix, filled with `data`
    #[inline]
    pub fn from_value(data: f64, (rows, cols): MatrixIndex) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        Self {
            data: MatrixData::Mutable(vec![data; rows * cols]),
            rows,
            cols,
        }
    }
    /// Constructs a new reference matrix by `data`
    #[inline]
    pub fn from_ref(data: &'a [f64], rows: usize) -> Self {
        assert!(rows > 0);
        assert!(data.len() % rows == 0);
        let cols = data.len() / rows;
        Self {
            data: MatrixData::Reference(data),
            rows,
            cols,
        }
    }
    /// Returns the underlying data (stored in column major).
    #[inline]
    pub fn data(&self) -> &[f64] {
        match self.data {
            MatrixData::Mutable(ref v) => v.as_slice(),
            MatrixData::Reference(v) => v,
        }
    }
    /// Returns the underlying data (stored in column major).
    /// If the matrix is a reference matrix, the matrix is transformed to a mutable matrix.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f64] {
        match self.data {
            MatrixData::Mutable(ref mut v) => return v.as_mut_slice(),
            _ => {
                self.to_mut();
                return self.data_mut();
            }
        };
    }
    /// Returns the number of rows in the matrix
    #[inline]
    pub fn nrow(&self) -> usize {
        self.rows
    }
    /// Returns the number of columns in the matrix
    #[inline]
    pub fn ncol(&self) -> usize {
        self.cols
    }
    /// Returns the dimensions of the matrix
    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    /// Returns an iterator on the row
    #[inline]
    pub fn row_iter(&self, row: usize) -> MatrixIterator {
        assert!(row < self.nrow());
        MatrixIterator {
            iter: self.data().iter().skip(row).step_by(self.nrow()),
            dim: self.dim(),
            step: self.nrow(),
            count: 0,
        }
    }
    /// Returns an iterator on the column
    #[allow(clippy::iter_skip_zero)]
    #[inline]
    pub fn col_iter(&self, col: usize) -> MatrixIterator {
        assert!(col < self.ncol());
        MatrixIterator {
            iter: self.data()[(self.nrow() * col)..(self.nrow() * (col + 1))]
                .iter()
                .skip(0)
                .step_by(1),
            dim: self.dim(),
            step: 1,
            count: 0,
        }
    }
    /// Resizes the matrix, without guaranteeing the preservation of any data.
    /// If the matrix is a reference matrix, the matrix is transformed to a mutable matrix.
    #[inline]
    pub fn resize(&mut self, (rows, cols): MatrixIndex) -> &Self {
        assert!(rows > 0 && cols > 0);
        let new_size = rows * cols;

        match self.data {
            MatrixData::Mutable(ref mut v) => {
                v.resize(new_size, 0.0);
            }
            _ => {
                self.to_mut();
                return self.resize((rows, cols));
            }
        }

        self.rows = rows;
        self.cols = cols;
        self
    }
    /// Calculates the reduced row echelon form of the matrix, in place.
    /// If the matrix is a reference matrix, the matrix is transformed to a mutable matrix.
    pub fn reduced_row_echelon_form(&mut self) {
        let index = |midx: MatrixIndex| midx.0 + midx.1 * self.rows;
        let data = match self.data {
            MatrixData::Mutable(ref mut v) => v.as_mut_slice(),
            _ => {
                self.to_mut();
                return self.reduced_row_echelon_form();
            }
        };

        let mut lead: usize = 0;

        for row in 0..self.rows {
            if self.cols <= lead {
                return;
            }

            let mut i: usize = row;

            while unsafe { *data.get_unchecked(index((i, lead))) == 0.0 } {
                i += 1;

                if i == self.rows {
                    i = row;
                    lead += 1;
                }

                if lead == self.cols {
                    return;
                }
            }

            // Swap rows i and row
            if i != row {
                let mut index_i = i;
                let mut index_row = row;
                for _ in 0..self.cols {
                    data.swap(index_i, index_row);
                    index_i += self.rows;
                    index_row += self.rows;
                }
            }

            // Divide ROW by lead, assuming all is 0 before lead
            let mut index_row = index((row, lead));
            let lead_value = unsafe { *data.get_unchecked(index_row) };
            if lead_value != 1.0 {
                unsafe {
                    *data.get_unchecked_mut(index_row) = 1.0;
                }
                index_row += self.rows;

                for _ in (lead + 1)..self.cols {
                    unsafe {
                        *data.get_unchecked_mut(index_row) /= lead_value;
                    }
                    index_row += self.rows;
                }
            }

            // Remove ROW from all other rows
            for j in 0..self.rows {
                if j == row {
                    continue;
                }

                let mut index_j = index((j, lead));
                let lead_multiplicator = unsafe { *data.get_unchecked(index_j) };
                if lead_multiplicator == 0.0 {
                    continue;
                }
                unsafe {
                    *data.get_unchecked_mut(index_j) = 0.0;
                }
                index_j += self.rows;
                let mut index_row = index((row, lead + 1));

                for _ in (lead + 1)..self.cols {
                    unsafe {
                        *data.get_unchecked_mut(index_j) -=
                            *data.get_unchecked(index_row) * lead_multiplicator;
                    }
                    index_j += self.rows;
                    index_row += self.rows;
                }
            }

            lead += 1;
        }
    }
    /// Returns the squared eculidean distance between the `row` and the slice `unit`
    #[inline]
    pub fn distance_to_row(&self, row: usize, unit: &[f64]) -> f64 {
        assert!(unit.len() == self.ncol());
        assert!(row < self.nrow());

        self.row_iter(row)
            .zip(unit.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
    }
    /// Performes the calculation of self * multiplicand, where self is a matrix A, and multiplicand
    /// is a vector.
    #[inline]
    pub fn prod_vec(&self, multiplicand: &[f64]) -> Vec<f64> {
        assert!(multiplicand.len() == self.ncol());
        let data = self.data();
        let mut prod = vec![0.0; self.nrow()];
        let mut index: usize = 0;

        for &mul in multiplicand.iter() {
            for pr in prod.iter_mut() {
                *pr += mul * data[index];
                index += 1;
            }
        }

        prod
    }
    /// Performes the calculation of matrices self * mat
    #[inline]
    pub fn mult(&self, mat: &Matrix) -> Matrix {
        assert!(self.ncol() == mat.nrow());
        let mut prod = Vec::<f64>::with_capacity(self.nrow() * mat.ncol());

        let mut index = 0usize;
        for _ in 0..mat.ncol() {
            prod.extend_from_slice(&self.prod_vec(&mat.data()[index..(index + mat.nrow())]));
            index += mat.nrow();
        }

        Matrix::new(&prod, self.nrow())
    }
}

impl<'a> Index<MatrixIndex> for Matrix<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, midx: MatrixIndex) -> &f64 {
        unsafe { self.data().get_unchecked(self.matrix_index(midx)) }
    }
}
impl<'a> IndexMut<MatrixIndex> for Matrix<'a> {
    #[inline]
    fn index_mut(&mut self, midx: MatrixIndex) -> &mut f64 {
        let idx = self.matrix_index(midx);
        match self.data {
            MatrixData::Mutable(ref mut v) => unsafe { v.get_unchecked_mut(idx) },
            _ => {
                self.to_mut();
                return self.index_mut(midx);
            }
        }
    }
}
impl<'a> Clone for Matrix<'a> {
    #[inline]
    fn clone(&self) -> Self {
        let slice = match self.data {
            MatrixData::Mutable(ref v) => v.as_slice(),
            MatrixData::Reference(v) => v,
        };

        Self {
            rows: self.rows,
            cols: self.cols,
            data: MatrixData::Mutable(slice.to_vec()),
        }
    }
}

pub struct MatrixIterator<'a> {
    iter: StepBy<Skip<Iter<'a, f64>>>,
    dim: MatrixIndex,
    step: usize,
    count: usize,
}
impl<'a> MatrixIterator<'a> {
    #[inline]
    pub fn dim(&self) -> MatrixIndex {
        if self.step == 1 {
            (self.dim.0, 1)
        } else {
            (1, self.dim.1)
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
            self.dim.0 - self.count
        } else {
            self.dim.1 - self.count
        }
    }
}
