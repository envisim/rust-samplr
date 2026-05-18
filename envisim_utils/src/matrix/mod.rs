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

//! Matrix representations
//!
//! Provides [`Matrix`] and [`MatrixRef`] as mutable and borrowed matrix representations.
//! Both are derived from [`MatrixBase`].

mod dims;

use std::num::NonZeroUsize;
use std::ops::{
    Index,
    IndexMut,
};

pub use dims::{
    Dimensions,
    MatrixCoord,
    MatrixDims,
};

use crate::number_traits::{
    Number,
    NumberFloat,
};
pub use crate::spatial::PointSet;

/// Data container trait
pub trait RawData: Sized {
    type Elem;
    #[must_use]
    fn data(&self) -> &[Self::Elem];
}

/// Base matrix representation
#[must_use]
#[derive(Debug, Clone)]
pub struct MatrixBase<T, N = <T as RawData>::Elem>
where
    T: RawData<Elem = N>,
{
    /// Data
    data: T,
    /// Matrix dimension
    dims: MatrixDims,
}

/// Owned matrix representation
pub type Matrix<N> = MatrixBase<OwnedMatrixData<N>>;
/// Borrowed matrix representation
pub type MatrixRef<'bdata, N> = MatrixBase<BorrowedMatrixData<'bdata, N>>;

impl<T, N> MatrixBase<T, N>
where
    T: RawData<Elem = N>,
{
    /// Constructs a borrowed matrix representation from a matrix.
    #[inline]
    pub fn to_matrixref(&self) -> MatrixRef<'_, N> {
        MatrixRef {
            data: self.internal_data().into(),
            dims: self.dims(),
        }
    }
    /// Constructs an owned matrix representation from a matrix. Copies the data.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<N>
    where
        N: Copy,
    {
        Matrix {
            data: self.internal_data().to_vec().into(),
            dims: self.dims(),
        }
    }
    /// Returns a reference to the underlying data
    #[must_use]
    #[inline]
    pub fn data(&self) -> &T { &self.data }
    /// Returns a reference to the underlying data as a slice.
    #[must_use]
    #[inline]
    fn internal_data(&self) -> &[N] { self.data().data() }
    /// Returns the element at a specific coordinate.
    /// Returns `None`  if the coordinates are invalid.
    #[must_use]
    #[inline]
    pub fn get<C>(&self, coord: C) -> Option<&N>
    where
        C: Into<MatrixCoord>,
        N: Copy,
    {
        let coord = coord.into();
        self.dims.contains(coord).then(|| &self[coord])
    }
    /// Returns an iterator of the elements in a row.
    /// Returns `None` if the row is invalid.
    #[must_use]
    #[inline]
    pub fn row_iter(&self, row: usize) -> Option<impl ExactSizeIterator<Item = &N>> {
        let nrow = self.nrow().get();
        self.dims()
            .contains_row(row)
            .then(|| self.internal_data()[row..].iter().step_by(nrow))
    }
    /// Returns an iterator of the elements in a column.
    /// Returns `None` if the column is invalid.
    #[must_use]
    #[inline]
    pub fn col_iter(&self, col: usize) -> Option<impl ExactSizeIterator<Item = &N>> {
        let nrow = self.nrow().get();
        let start = nrow * col;
        self.dims()
            .contains_col(col)
            .then(|| self.internal_data()[start..(start + nrow)].iter())
    }
    /// Multiplies the matrix by a column vector.
    /// Returns `None` if the column vector length does not match the number of columns in the
    /// matrix.
    #[must_use]
    #[inline]
    pub fn mul_vec(&self, rhs: &[N]) -> Option<Matrix<N>>
    where
        N: Number,
    {
        if self.ncol().get() != rhs.len() {
            return None;
        }
        let mut product = vec![N::zero(); self.nrow().get()];
        let mut index = 0;
        for mul in rhs {
            for pr in &mut product {
                *pr += *mul * self.internal_data()[index];
                index += 1;
            }
        }
        Matrix::new(product, self.nrow())
    }
    /// Multiplies the matrix by another matrix.
    /// Returns `None` if the number of columns in self does not match the number of rows in the
    /// other matrix.
    #[must_use]
    #[inline]
    pub fn mul_mat<T2>(&self, rhs: &MatrixBase<T2, N>) -> Option<Matrix<N>>
    where
        T2: RawData<Elem = N>,
        N: Number,
    {
        if self.ncol() != rhs.nrow() {
            return None;
        }
        let mut product = Vec::<N>::with_capacity(self.nrow().get() * rhs.ncol().get());
        let mut index = 0;
        // Multiply self by each column in rhs
        for _ in 0..rhs.ncol().get() {
            // Take rhs column
            let rhs_col = &rhs.internal_data()[index..(index + rhs.nrow().get())];
            // Result
            let temp_column_res = self.mul_vec(rhs_col)?;
            product.extend_from_slice(temp_column_res.internal_data());
            index += rhs.nrow().get();
        }
        Matrix::new(product, self.nrow())
    }
}

impl<T, N> Dimensions for MatrixBase<T, N>
where
    T: RawData<Elem = N>,
{
    /// Returns the matrix dimension.
    #[inline]
    fn dims(&self) -> MatrixDims { self.dims }
}
impl<T, N, I> Index<I> for MatrixBase<T, N>
where
    T: RawData<Elem = N>,
    I: Into<MatrixCoord>,
{
    type Output = T::Elem;
    /// # Panics
    /// If index is out of bounds.
    #[must_use]
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let index = index
            .into()
            .to_linear(self.dims)
            .expect("index to be valid for the matrix");
        &self.internal_data()[index]
    }
}
impl<T, N> From<&MatrixBase<T, N>> for Matrix<N>
where
    T: RawData<Elem = N>,
    N: Copy,
{
    #[inline]
    fn from(matrix: &MatrixBase<T, N>) -> Self { matrix.to_matrix() }
}
impl<'bdata, T, N> From<&'bdata MatrixBase<T, N>> for MatrixRef<'bdata, N>
where
    T: RawData<Elem = N>,
{
    #[inline]
    fn from(matrix: &'bdata MatrixBase<T, N>) -> Self { matrix.to_matrixref() }
}

impl<T, N> PointSet for MatrixBase<T, N>
where
    T: RawData<Elem = N>,
    N: Number,
{
    type N = N;
    /// Returns the number of rows in the matrix
    #[inline]
    fn size(&self) -> NonZeroUsize { self.dims.rows }
    /// Returns an iterator of the rows in the matrix.
    #[inline]
    fn id_iter(&self) -> impl Iterator<Item = usize> { 0..self.dims.rows.get() }
    /// Returns the number of columns in the matrix
    #[inline]
    fn dim(&self) -> NonZeroUsize { self.dims.cols }
    /// Returns true if `id` is contained within the matrix.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    fn exists(&self, row: usize) -> bool { row < self.dims.rows.get() }
    /// Returns the element at coordinates `(row, col)`.
    /// Panics on oob.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    fn coord(&self, row: usize, col: usize) -> N {
        let idx = row + col * self.dims.rows.get();
        self.internal_data()[idx]
    }
    /// Returns the element at coordinates `(row, col)`.
    /// Returns `None` if the coordinates are oob.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    fn try_coord(&self, row: usize, col: usize) -> Option<N> { self.get((row, col)).copied() }
    /// Returns the squared euclidean distance between rows `id_a` and `id_b`.
    /// Panics on oob.
    #[inline]
    fn sq_distance_between(&self, id_a: usize, id_b: usize) -> N {
        if id_a == id_b {
            return N::zero();
        }
        let mut idx = id_a.min(id_b);
        let idx_diff = id_a.abs_diff(id_b);
        let mut sum = N::zero();
        for _ in 0..self.ncol().get() {
            let diff = self.internal_data()[idx] - self.internal_data()[idx + idx_diff];
            sum += diff * diff;
            idx += self.nrow().get();
        }
        sum
    }
    /// Returns the squared euclidean distance between rows `id_a` and `id_b`.
    /// Returns `None` if any row is oob.
    #[inline]
    fn try_sq_distance_between(&self, id_a: usize, id_b: usize) -> Option<N> {
        (self.exists(id_a) && self.exists(id_b)).then(|| self.sq_distance_between(id_a, id_b))
    }
}

impl<N> Matrix<N> {
    /// Constructs a new owned matrix representation from some data vector.
    /// Returns `None` if the rows are not a divisor of the data length.
    #[inline]
    pub fn new<NZ>(data: Vec<N>, rows: NZ) -> Option<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let rows = rows.try_into().ok()?;
        let dims = MatrixDims::from_row_count(data.len(), rows)?;
        let data = OwnedMatrixData::new(data);
        Some(Self { data, dims })
    }
    /// Constructs a new owned matrix representation filled with some value `data`.
    #[inline]
    pub fn from_value<D>(data: N, dims: D) -> Self
    where
        N: Copy,
        D: Into<MatrixDims>,
    {
        let dims: MatrixDims = dims.into();
        Self {
            data: OwnedMatrixData::new(vec![data; dims.len().get()]),
            dims,
        }
    }
    /// Returns a mutable reference to the element at a specific coordinate.
    /// Returns `None`  if the coordinates are invalid.
    #[must_use]
    #[inline]
    pub fn get_mut<C>(&mut self, coord: C) -> Option<&mut N>
    where
        C: Into<MatrixCoord>,
    {
        let coord = coord.into();
        self.dims.contains(coord).then(|| &mut self[coord])
    }
    /// Returns a mutable iterator of the elements in a row.
    /// Returns `None` if the row is invalid.
    #[must_use]
    #[inline]
    pub fn row_iter_mut(&mut self, row: usize) -> Option<impl ExactSizeIterator<Item = &mut N>> {
        let nrow = self.nrow().get();
        self.dims()
            .contains_row(row)
            .then(|| self.data.data[row..].iter_mut().step_by(nrow))
    }
    /// Returns a mutable iterator of the elements in a column.
    /// Returns `None` if the column is invalid.
    #[must_use]
    #[inline]
    pub fn col_iter_mut(&mut self, col: usize) -> Option<impl ExactSizeIterator<Item = &mut N>> {
        let nrow = self.nrow().get();
        let start = nrow * col;
        self.dims()
            .contains_col(col)
            .then(|| self.data.data[start..(start + nrow)].iter_mut())
    }
    /// Resizes the matrix.
    /// Does not respect data on expansion.
    #[inline]
    pub fn resize<D>(&mut self, dims: D)
    where
        N: Copy + num_traits::Zero,
        D: Into<MatrixDims>,
    {
        let dims: MatrixDims = dims.into();
        if dims == self.dims() {
            return;
        }
        self.data.data.resize(dims.len().get(), N::zero());
        self.dims = dims;
    }
    /// Calculates the reduced row echelon form of the matrix, in place.
    #[inline]
    pub fn reduced_row_echelon_form(&mut self)
    where
        N: NumberFloat,
    {
        let dims = self.dims();
        let index = |row, col| row + col * dims.rows.get();
        let data = &mut self.data.data;

        let mut lead: usize = 0;

        // We can skip som tolerance on equality checks, as we can guarantee that (some) are exactly
        // 0.0 or 1.0
        for row in 0..dims.rows.get() {
            if dims.cols.get() <= lead {
                return;
            }

            let mut i: usize = row;

            while data[index(i, lead)] == N::zero() {
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
            {
                let mut index_row = index(row, lead);
                let lead_value = data[index_row];
                if lead_value != N::one() {
                    data[index_row] = N::one();
                    index_row += dims.rows.get();

                    for _ in (lead + 1)..dims.cols.get() {
                        data[index_row] /= lead_value;
                        index_row += dims.rows.get();
                    }
                }
            }

            // Remove ROW from all other rows
            for j in 0..dims.rows.get() {
                if j == row {
                    continue;
                }

                let mut index_j = index(j, lead);

                let lead_multiplicator = data[index_j];
                if lead_multiplicator == N::zero() {
                    continue;
                }

                data[index_j] = N::zero();
                index_j += dims.rows.get();
                let mut index_row = index(row, lead + 1);

                for _ in (lead + 1)..dims.cols.get() {
                    let delta = data[index_row] * lead_multiplicator;
                    data[index_j] -= delta;
                    index_j += dims.rows.get();
                    index_row += dims.rows.get();
                }
            }

            lead += 1;
        }
    }
}

impl<'bdata, N> MatrixRef<'bdata, N> {
    #[inline]
    pub fn new<NZ>(data: &'bdata [N], rows: NZ) -> Option<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        let rows = rows.try_into().ok()?;
        let dims = MatrixDims::from_row_count(data.len(), rows)?;
        let data = BorrowedMatrixData::new(data);
        Some(Self { data, dims })
    }
}

impl<N, I> IndexMut<I> for Matrix<N>
where
    I: Into<MatrixCoord>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = index
            .into()
            .to_linear(self.dims())
            .expect("index to be valid for the matrix");
        &mut self.data.data[index]
    }
}

#[must_use]
#[derive(Debug, Clone)]
pub struct OwnedMatrixData<N> {
    /// Internal data vector
    data: Vec<N>,
}
impl<N> OwnedMatrixData<N> {
    /// Constructs a new owned matrix data
    #[inline]
    pub fn new(data: Vec<N>) -> Self { Self { data } }
}

impl<N> RawData for OwnedMatrixData<N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { &self.data }
}
impl<N> From<&BorrowedMatrixData<'_, N>> for OwnedMatrixData<N>
where
    N: Copy,
{
    #[inline]
    fn from(data: &BorrowedMatrixData<N>) -> Self { Self::new(data.data().to_vec()) }
}
impl<N> From<&[N]> for OwnedMatrixData<N>
where
    N: Copy,
{
    #[inline]
    fn from(data: &[N]) -> Self { Self::new(data.to_vec()) }
}
impl<N> From<Vec<N>> for OwnedMatrixData<N> {
    #[inline]
    fn from(data: Vec<N>) -> Self { Self::new(data) }
}

#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct BorrowedMatrixData<'bdata, N> {
    /// Internal data reference
    data: &'bdata [N],
}
impl<'bdata, N> BorrowedMatrixData<'bdata, N> {
    /// Constructs a new borrowed matrix data
    #[inline]
    pub fn new(data: &'bdata [N]) -> Self { Self { data } }
}

impl<N> RawData for BorrowedMatrixData<'_, N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.data }
}
impl<'borrow, N> From<&'borrow OwnedMatrixData<N>> for BorrowedMatrixData<'borrow, N> {
    #[inline]
    fn from(data: &'borrow OwnedMatrixData<N>) -> Self { Self::new(data.data()) }
}
impl<'bdata, N> From<&'bdata [N]> for BorrowedMatrixData<'bdata, N> {
    #[inline]
    fn from(data: &'bdata [N]) -> Self { Self::new(data) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    /// Creates a 3x2 matrix in column-major order:
    /// [ 1, 4 ]
    /// [ 2, 5 ]
    /// [ 3, 6 ]
    fn setup_matrix() -> Matrix<f64> {
        Matrix::new(
            vec![
                1.0, 2.0, 3.0, // dim 0
                4.0, 5.0, 6.0, // dim 1
            ],
            nz(3),
        )
        .unwrap()
    }

    #[test]
    fn test_matrix_construction() {
        // Valid construction
        let m = Matrix::new(vec![1, 2, 3, 4], nz(2));
        assert!(m.is_some());

        // Invalid: 5 elements cannot form a matrix with 2 rows
        let m_invalid = Matrix::new(vec![1, 2, 3, 4, 5], nz(2));
        assert!(m_invalid.is_none());

        // From value
        let m_val = Matrix::from_value(10, MatrixDims::new(nz(2), nz(2)));
        assert_eq!(m_val.data.data(), &[10, 10, 10, 10]);
    }

    #[test]
    fn test_accessors() {
        let m = setup_matrix();
        assert_eq!(m.nrow(), nz(3));
        assert_eq!(m.ncol(), nz(2));
        assert_eq!(m.dims().rows, nz(3));

        // get() method
        assert_eq!(m.get((0, 0)), Some(&1.0));
        assert_eq!(m.get((2, 1)), Some(&6.0));
        assert_eq!(m.get((3, 0)), None); // Out of bounds
    }

    #[test]
    fn test_indexing() {
        let mut m = setup_matrix();

        // Index read
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(0, 1)], 4.0);

        // IndexMut write
        m[(0, 1)] = 10.0;
        assert_eq!(m[(0, 1)], 10.0);
    }

    #[test]
    fn test_conversions() {
        let m = setup_matrix();

        // To MatrixRef
        let m_ref: MatrixRef<f64> = m.to_matrixref();
        assert_eq!(m_ref[(0, 0)], 1.0);

        // To Matrix (Clone/Copy)
        let m_owned = m_ref.to_matrix();
        assert_eq!(m_owned[(2, 1)], 6.0);
    }

    #[test]
    fn test_iterators() {
        let m = setup_matrix();

        // Row Iterator for row 1: [2.0, 5.0]
        let mut row_iter = m.row_iter(1).unwrap();
        assert_eq!(row_iter.len(), 2);
        assert_eq!(row_iter.next(), Some(&2.0));
        assert_eq!(row_iter.next(), Some(&5.0));
        assert_eq!(row_iter.next(), None);

        // Column Iterator for col 1: [4.0, 5.0, 6.0]
        let mut col_iter = m.col_iter(1).unwrap();
        assert_eq!(col_iter.len(), 3);
        assert_eq!(col_iter.next(), Some(&4.0));
        assert_eq!(col_iter.next(), Some(&5.0));
        assert_eq!(col_iter.next(), Some(&6.0));
        assert_eq!(col_iter.next(), None);

        // Invalid indices
        assert!(m.row_iter(3).is_none());
        assert!(m.col_iter(2).is_none());
    }

    #[test]
    fn test_point_set_trait() {
        let m = setup_matrix();

        // Basic trait methods
        assert_eq!(PointSet::size(&m), nz(3));
        assert_eq!(PointSet::dim(&m), nz(2));
        assert!(m.exists(2));
        assert!(!m.exists(3));

        // Coordinate access
        assert_eq!(m.coord(1, 1), 5.0);
        assert_eq!(m.try_coord(2, 0), Some(3.0));
        assert_eq!(m.try_coord(3, 0), None);
    }

    #[test]
    fn test_distance_calculations() {
        // Matrix:
        // R0: [1, 4]
        // R1: [2, 5]
        // Distance R0 to R1: (1-2)^2 + (4-5)^2 = 1 + 1 = 2
        let m = setup_matrix();

        let dist_sq = m.sq_distance_between(0, 1);
        assert_eq!(dist_sq, 2.0);

        let same_dist = m.sq_distance_between(2, 2);
        assert_eq!(same_dist, 0.0);

        assert!(m.try_sq_distance_between(0, 3).is_none());
    }

    #[test]
    fn test_resize() {
        let mut m = setup_matrix(); // 3x2
        let new_dims = MatrixDims::new(nz(2), nz(2)); // 2x2

        m.resize(new_dims);
        assert_eq!(m.nrow(), nz(2));
        assert_eq!(m.ncol(), nz(2));
        // Verify data was truncated/kept according to layout
        assert_eq!(m.data.data().len(), 4);
    }

    #[test]
    fn test_linear_algebra_stubs() {
        let m = setup_matrix(); // 3x2

        // mul_vec: rhs must match ncol (2)
        let vec_good = vec![1.0, 1.0];
        let vec_bad = vec![1.0, 1.0, 1.0];

        assert!(m.mul_vec(&vec_good).is_some());
        assert!(m.mul_vec(&vec_bad).is_none());

        // mul_mat: rhs rows (2) must match lhs cols (2)
        let rhs_good = Matrix::from_value(1.0, MatrixDims::new(nz(2), nz(4)));
        let rhs_bad = Matrix::from_value(1.0, MatrixDims::new(nz(3), nz(4)));

        assert!(m.mul_mat(&rhs_good).is_some());
        assert!(m.mul_mat(&rhs_bad).is_none());
    }
}
