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

use std::iter::FusedIterator;
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
use num_traits::{
    ConstOne,
    ConstZero,
    Float,
};

use crate::utils::{
    Epsilon,
    Number,
    NumberFloat,
};
pub use crate::utils::{
    PointSet,
    SliceView,
    SliceViewMut,
};

/// Base matrix representation
#[must_use]
#[derive(Debug, Clone)]
pub struct MatrixBase<T> {
    /// Data
    data: T,
    /// Matrix dimension
    dims: MatrixDims,
}

/// Owned matrix representation
pub type Matrix<N> = MatrixBase<Vec<N>>;
/// Borrowed matrix representation
pub type MatrixRef<'bdata, N> = MatrixBase<&'bdata [N]>;

impl<T> MatrixBase<T> {
    /// Constructs a new owned matrix representation from some data vector.
    /// Returns `None` if the rows are not a divisor of the data length.
    #[inline]
    pub fn new<NZ>(data: T, rows: NZ) -> Option<Self>
    where
        T: SliceView,
        NZ: TryInto<NonZeroUsize>,
    {
        let rows = rows.try_into().ok()?;
        let dims = MatrixDims::from_row_count(data.data().len(), rows)?;
        Some(Self { data, dims })
    }
    /// Constructs a new owned matrix representation filled with some value `data`.
    #[inline]
    pub fn from_value<D>(value: T::Elem, dims: D) -> Self
    where
        T: SliceView + From<Vec<T::Elem>>,
        T::Elem: Copy,
        D: Into<MatrixDims>,
    {
        let dims: MatrixDims = dims.into();
        let data = vec![value; dims.len().get()];
        Self {
            data: data.into(),
            dims,
        }
    }
    /// Constructs a new identity matrix
    #[inline]
    pub fn new_identity<NZ>(rows: NZ) -> Option<Self>
    where
        T: SliceView + From<Vec<T::Elem>>,
        T::Elem: ConstZero + ConstOne + Copy,
        NZ: TryInto<NonZeroUsize>,
    {
        let rows = rows.try_into().ok()?;
        let dims = MatrixDims::new(rows, rows);
        let mut data = vec![T::Elem::ZERO; dims.len().get()];
        for e in data.iter_mut().step_by(rows.get() + 1) {
            *e = T::Elem::ONE;
        }
        Some(Self {
            data: data.into(),
            dims,
        })
    }
    /// Constructs a borrowed matrix representation from a matrix.
    #[inline]
    pub fn to_matrixref(&self) -> MatrixRef<'_, T::Elem>
    where
        T: SliceView,
    {
        MatrixRef {
            data: self.data.data(),
            dims: self.dims(),
        }
    }
    /// Constructs an owned matrix representation from a matrix. Copies the data.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<T::Elem>
    where
        T: SliceView,
        T::Elem: Copy,
    {
        Matrix {
            data: self.data.data().to_vec(),
            dims: self.dims(),
        }
    }
    /// Returns a reference to the underlying data
    #[must_use]
    #[inline]
    pub fn data(&self) -> &T { &self.data }
    /// Returns the element at a specific coordinate.
    /// Returns `None`  if the coordinates are invalid.
    #[must_use]
    #[inline]
    pub fn get<C>(&self, coord: C) -> Option<&T::Elem>
    where
        T: SliceView,
        C: Into<MatrixCoord>,
    {
        let coord = coord.into();
        self.dims.contains(coord).then(|| &self[coord])
    }
    /// Returns a mutable reference to the element at a specific coordinate.
    /// Returns `None`  if the coordinates are invalid.
    #[must_use]
    #[inline]
    pub fn get_mut<C>(&mut self, coord: C) -> Option<&mut T::Elem>
    where
        T: SliceViewMut,
        C: Into<MatrixCoord>,
    {
        let coord = coord.into();
        self.dims.contains(coord).then(|| &mut self[coord])
    }
    /// Swaps the element at `coord_a` with the element at `coord_b`.
    #[inline]
    pub fn swap<CA, CB>(&mut self, coord_a: CA, coord_b: CB) -> Option<()>
    where
        T: SliceViewMut,
        T::Elem: Copy,
        CA: Into<MatrixCoord>,
        CB: Into<MatrixCoord>,
    {
        let idx_a = coord_a.into().to_linear(self.dims)?;
        let idx_b = coord_b.into().to_linear(self.dims)?;
        self.data.data_mut().swap(idx_a, idx_b);
        Some(())
    }
    /// Returns an iterator of the elements in a row.
    /// Returns `None` if the row is invalid.
    #[must_use]
    #[inline]
    pub fn row_iter(
        &self,
        row: usize,
    ) -> Option<impl ExactSizeIterator<Item = &T::Elem> + DoubleEndedIterator + Clone>
    where
        T: SliceView,
    {
        let nrow = self.nrow().get();
        self.dims()
            .contains_row(row)
            .then(|| self.data.data()[row..].iter().step_by(nrow))
    }
    /// Returns a mutable iterator of the elements in a row.
    /// Returns `None` if the row is invalid.
    #[must_use]
    #[inline]
    pub fn row_iter_mut(
        &mut self,
        row: usize,
    ) -> Option<impl ExactSizeIterator<Item = &mut T::Elem> + DoubleEndedIterator>
    where
        T: SliceViewMut,
    {
        let nrow = self.nrow().get();
        self.dims()
            .contains_row(row)
            .then(|| self.data.data_mut()[row..].iter_mut().step_by(nrow))
    }
    /// Returns an iterator of the elements in a column.
    /// Returns `None` if the column is invalid.
    #[must_use]
    #[inline]
    pub fn col_iter(
        &self,
        col: usize,
    ) -> Option<impl ExactSizeIterator<Item = &T::Elem> + DoubleEndedIterator + FusedIterator + Clone>
    where
        T: SliceView,
    {
        let nrow = self.nrow().get();
        let start = nrow * col;
        self.dims()
            .contains_col(col)
            .then(|| self.data.data()[start..(start + nrow)].iter())
    }
    /// Returns a mutable iterator of the elements in a column.
    /// Returns `None` if the column is invalid.
    #[must_use]
    #[inline]
    pub fn col_iter_mut(
        &mut self,
        col: usize,
    ) -> Option<impl ExactSizeIterator<Item = &mut T::Elem> + DoubleEndedIterator + FusedIterator>
    where
        T: SliceViewMut,
    {
        let nrow = self.nrow().get();
        let start = nrow * col;
        self.dims()
            .contains_col(col)
            .then(|| self.data.data_mut()[start..(start + nrow)].iter_mut())
    }
    /// Multiplies the matrix by a column vector.
    /// Returns `None` if the column vector length does not match the number of columns in the
    /// matrix.
    #[must_use]
    #[inline]
    pub fn mul_vec(&self, rhs: &[T::Elem]) -> Option<Matrix<T::Elem>>
    where
        T: SliceView,
        T::Elem: Number,
    {
        if self.ncol().get() != rhs.len() {
            return None;
        }
        let mut product = vec![T::Elem::ZERO; self.nrow().get()];
        let mut index = 0;
        for mul in rhs {
            for pr in &mut product {
                *pr += *mul * self.data.data()[index];
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
    pub fn mul_mat<T2>(&self, rhs: &MatrixBase<T2>) -> Option<Matrix<T::Elem>>
    where
        T: SliceView,
        T::Elem: Number,
        T2: SliceView<Elem = T::Elem>,
    {
        if self.ncol() != rhs.nrow() {
            return None;
        }
        let mut product = Vec::<T::Elem>::with_capacity(self.nrow().get() * rhs.ncol().get());
        let mut index = 0;
        // Multiply self by each column in rhs
        for _ in 0..rhs.ncol().get() {
            // Take rhs column
            let rhs_col = &rhs.data.data()[index..(index + rhs.nrow().get())];
            // Result
            let temp_column_res = self.mul_vec(rhs_col)?;
            product.extend_from_slice(temp_column_res.data.data());
            index += rhs.nrow().get();
        }
        Matrix::new(product, self.nrow())
    }
    /// Calculates the inverse of `self` using LU decomposition.
    ///
    /// Returns `None` if `self` isn't square, or if `self` is not invertible (as defined by `eps`).
    #[expect(clippy::many_single_char_names, reason = "only within small scope")]
    #[must_use]
    #[inline]
    pub fn inverse<E>(&self, eps: E) -> Option<Matrix<T::Elem>>
    where
        T: SliceView,
        T::Elem: NumberFloat,
        E: TryInto<Epsilon<T::Elem>>,
    {
        // Non-square
        if !self.dims().is_square() {
            return None;
        }
        let eps = eps.try_into().ok()?;
        let nrow = self.nrow().get();

        if nrow == 2 {
            const NZ2: NonZeroUsize = NonZeroUsize::new(2).expect("2 > 0");
            // ab cd => 02 13
            #[expect(clippy::unreachable, reason = "panic implies bug")]
            let &[a, c, b, d] = self.data.data() else {
                unreachable!("2x2 = 4")
            };
            // ad-bc
            let det = a * d - b * c;
            if eps.is_zero(det) {
                return None;
            };
            // d -c -b a
            let inv = vec![d / det, -c / det, -b / det, a / det];
            return MatrixBase::new(inv, NZ2);
        } else if nrow == 3 {
            const NZ3: NonZeroUsize = NonZeroUsize::new(3).expect("3 > 0");
            // abc def ghi => 036 147 258
            #[expect(clippy::unreachable, reason = "panic implies bug")]
            let &[a, d, g, b, e, h, c, f, i] = self.data.data() else {
                unreachable!("3x3=9")
            };
            let mut inv = vec![
                e * i - f * h,
                -(d * i - f * g),
                d * h - e * g, // ABC
                -(b * i - c * h),
                a * i - c * g,
                -(a * h - b * g), // DEF
                b * f - c * e,
                -(a * f - c * d),
                a * e - b * d, // GHI
            ];
            let det = a * inv[0] + b * inv[1] + c * inv[2]; // aA + bB +cC
            if eps.is_zero(det) {
                return None;
            };
            for v in &mut inv {
                *v /= det;
            }
            return MatrixBase::new(inv, NZ3);
        }

        let (lu, p) = {
            let mut lu = self.to_matrix();
            let p = lu.lu_decomposition(eps)?;
            (lu, p)
        };

        let mut inv = MatrixBase::from_value(T::Elem::ZERO, self.dims());

        // Solve for each column of the identity mat
        for col in 0..nrow {
            // Solve LY = P
            for row in 0..nrow {
                let mut sum = if p[row] == col {
                    T::Elem::ONE
                } else {
                    T::Elem::ZERO
                };
                for k in 0..row {
                    sum -= lu[(row, k)] * inv[(k, col)];
                }
                inv[(row, col)] = sum;
            }

            // Solve UX =  Y for X
            for row in (0..nrow).rev() {
                let mut sum = inv[(row, col)];
                for k in (row + 1)..nrow {
                    sum -= lu[(row, k)] * inv[(k, col)];
                }
                inv[(row, col)] = sum / lu[(row, row)];
            }
        }

        Some(inv)
    }
    /// Calculates the reduced row echelon form of the matrix, in place.
    #[inline]
    pub fn reduced_row_echelon_form(&mut self)
    where
        T: SliceViewMut,
        T::Elem: NumberFloat,
    {
        let dims = self.dims();
        let index = |row, col| row + col * dims.rows.get();
        let data = self.data.data_mut();

        let mut lead: usize = 0;

        // We can skip som tolerance on equality checks, as we can guarantee that (some) are exactly
        // 0.0 or 1.0
        for row in 0..dims.rows.get() {
            if dims.cols.get() <= lead {
                return;
            }

            let mut i: usize = row;

            while data[index(i, lead)] == T::Elem::ZERO {
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
                if lead_value != T::Elem::ONE {
                    data[index_row] = T::Elem::ONE;
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
                if lead_multiplicator == T::Elem::ZERO {
                    continue;
                }

                data[index_j] = T::Elem::ZERO;
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
    /// In-place LUP decomposition.
    ///
    /// Decomposes $A$ (`self`) into a lower $L$ and an upper $U$ matrix, returning a permutation
    /// vector $p$ such that $PA = LU$. Here, the permutatio vector $p$ represents the column
    /// indices of the non-zero elements of $P$.
    ///
    /// Returns `None` if $A$ isn't square, or if $A$ is singular (as defined by `eps`).
    #[expect(clippy::missing_panics_doc, reason = "panic implies bug")]
    #[must_use]
    #[inline]
    pub fn lu_decomposition<E>(&mut self, eps: E) -> Option<Box<[usize]>>
    where
        T: SliceViewMut,
        T::Elem: NumberFloat,
        E: TryInto<Epsilon<T::Elem>>,
    {
        // Non-square
        if !self.dims().is_square() {
            return None;
        }
        let eps = eps.try_into().ok()?;
        let nrows = self.nrow().get();
        let mut p: Box<[usize]> = (0..nrows).collect::<Vec<usize>>().into_boxed_slice();

        for row in 0..nrows {
            // Find pivot_row, the row with the highest value in the row-column
            let mut pivot_row = row;
            let mut max_val = T::Elem::ZERO;
            for row_b in row..nrows {
                let val = Float::abs(self[(row_b, row)]);
                if !Float::is_finite(val) {
                    // val is infinite or nan
                    return None;
                }
                if max_val < val {
                    max_val = val;
                    pivot_row = row_b;
                }
            }

            // Singular matrix
            if eps.is_zero(max_val) {
                return None;
            }

            // Swap rows
            if pivot_row != row {
                p.swap(row, pivot_row);
                for col in 0..nrows {
                    // guaranteed: row < pivot_row
                    let index_r = MatrixCoord::new(row, col).to_linear_unchecked(self.dims());
                    let index_p = index_r + (pivot_row - row);
                    self.data.data_mut().swap(index_r, index_p);
                }
            }

            // pivot_row is now row
            let pivot_val = self[(row, row)];
            for v in self.col_iter_mut(row).expect("row to exist").skip(row + 1) {
                *v /= pivot_val;
            }

            for col in (row + 1)..nrows {
                let akj = self[(row, col)];
                for row_b in (row + 1)..nrows {
                    let m = self[(row_b, row)] * akj;
                    self[(row_b, col)] -= m;
                }
            }
        }

        Some(p)
    }
}

impl<N> Matrix<N> {
    /// Resizes the matrix.
    /// Does not respect data on expansion.
    #[inline]
    pub fn resize<D>(&mut self, dims: D)
    where
        N: Copy + num_traits::ConstZero,
        D: Into<MatrixDims>,
    {
        let dims: MatrixDims = dims.into();
        if dims == self.dims() {
            return;
        }
        self.data.resize(dims.len().get(), N::ZERO);
        self.dims = dims;
    }
}

impl<T> Dimensions for MatrixBase<T> {
    /// Returns the matrix dimension.
    #[inline]
    fn dims(&self) -> MatrixDims { self.dims }
}

impl<T, I> Index<I> for MatrixBase<T>
where
    T: SliceView,
    I: Into<MatrixCoord>,
{
    type Output = T::Elem;
    /// # Panics
    /// If index is out of bounds.
    #[must_use]
    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let index = index.into().to_linear_unchecked(self.dims);
        &self.data.data()[index]
    }
}
impl<T, I> IndexMut<I> for MatrixBase<T>
where
    T: SliceViewMut,
    I: Into<MatrixCoord>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = index.into().to_linear_unchecked(self.dims());
        &mut self.data.data_mut()[index]
    }
}

impl<T> From<&MatrixBase<T>> for Matrix<T::Elem>
where
    T: SliceView,
    T::Elem: Copy,
{
    #[inline]
    fn from(matrix: &MatrixBase<T>) -> Self { matrix.to_matrix() }
}
impl<'bdata, T> From<&'bdata MatrixBase<T>> for MatrixRef<'bdata, T::Elem>
where
    T: SliceView,
{
    #[inline]
    fn from(matrix: &'bdata MatrixBase<T>) -> Self { matrix.to_matrixref() }
}

impl<T> PointSet for MatrixBase<T>
where
    T: SliceView,
    T::Elem: Number,
{
    type Value = T::Elem;
    type Id = usize;
    /// Returns the number of rows in the matrix
    #[inline]
    fn len(&self) -> NonZeroUsize { self.dims.rows }
    /// Returns an iterator of the rows in the matrix.
    #[inline]
    fn ids(&self) -> impl ExactSizeIterator<Item = Self::Id> + Clone { 0..self.dims.rows.get() }
    /// Returns the number of columns in the matrix
    #[inline]
    fn dimensions(&self) -> NonZeroUsize { self.dims.cols }
    /// Returns true if `id` is contained within the matrix.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    fn contains(&self, row: usize) -> bool { row < self.dims.rows.get() }
    /// Returns the element at coordinates `(row, col)`.
    /// Returns `None` if the coordinates are oob.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    fn coord(&self, row: usize, col: usize) -> Option<&Self::Value> { self.get((row, col)) }
    /// Returns the element at coordinates `(row, col)`.
    /// Panics on oob.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[inline]
    unsafe fn coord_unchecked(&self, row: usize, col: usize) -> &Self::Value {
        let idx = row + col * self.dims.rows.get();
        &self.data.data()[idx]
    }
    /// Returns an iterator over the coords of `row`, or `None` if `row` does not exist.
    #[expect(clippy::renamed_function_params, reason = "a matrix has rows, not ids")]
    #[must_use]
    #[inline]
    fn coords(
        &self,
        row: usize,
    ) -> Option<impl ExactSizeIterator<Item = &Self::Value> + DoubleEndedIterator + Clone> {
        self.row_iter(row)
    }
    #[must_use]
    #[inline]
    fn iter(
        &self,
    ) -> impl ExactSizeIterator<
        Item = impl ExactSizeIterator<Item = (Self::Id, usize, &Self::Value)>
               + DoubleEndedIterator
               + Clone,
    > + Clone {
        self.ids().map(move |id| {
            self.data.data()[id..]
                .iter()
                .step_by(self.dims.rows.get())
                .enumerate()
                .map(move |(c, v)| (id, c, v))
        })
    }
    #[must_use]
    #[inline]
    fn columns(
        &self,
    ) -> impl ExactSizeIterator<
        Item = impl ExactSizeIterator<Item = (Self::Id, usize, &Self::Value)> + Clone,
    > + DoubleEndedIterator
    + Clone {
        let mut start = 0;
        (0..self.dimensions().get()).map(move |c| {
            let end = start + self.dims.rows.get();
            let range = start..end;
            start = end;
            self.data.data()[range]
                .iter()
                .enumerate()
                .map(move |(id, v)| (id, c, v))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    /// Creates a 3x2 matrix
    /// [ 1 4
    ///   2 5
    ///   3 6 ]
    fn setup_matrix() -> Matrix<f64> {
        Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], nz(3)).unwrap()
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
        assert_eq!(PointSet::len(&m), nz(3));
        assert_eq!(PointSet::dimensions(&m), nz(2));
        assert!(m.contains(2));
        assert!(!m.contains(3));

        // Coordinate access
        assert_eq!(unsafe { m.coord_unchecked(1, 1) }, &5.0);
        assert_eq!(m.coord(2, 0), Some(&3.0));
        assert_eq!(m.coord(3, 0), None);
    }

    #[test]
    fn test_distance_calculations() {
        // Matrix:
        // R0: [1, 4]
        // R1: [2, 5]
        // Distance R0 to R1: (1-2)^2 + (4-5)^2 = 1 + 1 = 2
        let m = setup_matrix();

        let dist_sq = m.sq_distance_between(0, 1);
        assert_eq!(dist_sq, Some(2.0));

        let same_dist = m.sq_distance_between(2, 2);
        assert_eq!(same_dist, Some(0.0));

        assert!(m.sq_distance_between(0, 3).is_none());
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

    #[test]
    fn test_lu_decomposition() {
        // 3x3 matrix
        // [ 1 0 5
        //   2 1 6
        //   3 4 0 ]
        let mut valid_matrix = Matrix::new(vec![1., 2., 3., 0., 1., 4., 5., 6., 0.], 3).unwrap();

        let lu_res = valid_matrix.lu_decomposition(Epsilon::default());
        assert_eq!(lu_res.unwrap(), [2usize, 1, 0].into());
        assert_mat!(
            valid_matrix,
            Matrix::new(
                vec![3., 0.6667, 0.3333, 4.0, -1.6667, 0.8, 0.0, 6.0, 0.2],
                3
            )
            .unwrap(),
            Epsilon::new(1e-4).unwrap()
        );

        // Singular matrix (dependent columns)
        // [ 1 2 3
        //   1 2 3
        //   1 2 3 ]
        let mut singular_matrix = Matrix::new(vec![1., 2., 3., 1., 2., 3., 1., 2., 3.], 3).unwrap();

        let lu_singular = singular_matrix.lu_decomposition(Epsilon::default());
        assert!(lu_singular.is_none());
    }

    #[test]
    fn test_inverse_2x2() {
        // Matrix:
        // [ 4 3
        //   3 2 ]
        let m = Matrix::new(vec![4., 3., 3., 2.], 2).unwrap();

        // Expected Inverse:
        // [ -2  3
        //    3 -4 ]
        let expected = Matrix::new(vec![-2., 3., 3., -4.], 2).unwrap();

        let inv = m
            .inverse(Epsilon::default())
            .expect("Failed to invert 2x2 matrix");
        assert_mat!(inv, expected);
    }

    #[test]
    fn test_inverse_3x3() {
        // Matrix:
        // [ 1 2 3
        //   0 1 4
        //   5 6 0 ]
        let m = Matrix::new(vec![1., 0., 5., 2., 1., 6., 3., 4., 0.], 3).unwrap();

        // Inverse:
        // [-24  18  5
        //   20 -15 -4
        //   -5   4  1 ]
        let expected = Matrix::new(vec![-24., 20., -5., 18., -15., 4., 5., -4., 1.], 3).unwrap();

        let inv = m.inverse(Epsilon::default()).unwrap();
        assert_mat!(inv, expected);
    }

    #[test]
    fn test_inverse_4x4() {
        // Lower triangular matrix
        // [ 1 0 0 0
        //   2 1 0 0
        //   3 2 1 0
        //   4 3 2 1 ]
        let m = Matrix::new(
            vec![
                1., 2., 3., 4., 0., 1., 2., 3., 0., 0., 1., 2., 0., 0., 0., 1.,
            ],
            4,
        )
        .unwrap();

        // Inverse
        // [ 1  0  0  0
        //  -2  1  0  0
        //   1 -2  1  0
        //   0  1 -2  1 ]
        let expected = Matrix::new(
            vec![
                1., -2., 1., 0., 0., 1., -2., 1., 0., 0., 1., -2., 0., 0., 0., 1.,
            ],
            4,
        )
        .unwrap();

        let inv = m.inverse(Epsilon::default()).unwrap();
        assert_mat!(inv, expected);
    }

    #[test]
    fn test_inverse_singular() {
        // All elements are 1.0, making the matrix singular (determinant = 0)
        let m = Matrix::new(vec![1.0, 1.0, 1.0, 1.0], 2).unwrap();
        let inv = m.inverse(Epsilon::default());
        assert!(inv.is_none());
    }
}
