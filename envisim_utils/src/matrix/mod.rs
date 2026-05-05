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

mod dims;

use std::num::NonZeroUsize;
use std::ops::Index;

pub use dims::{
    MatrixCoord,
    MatrixDims,
};

use crate::number_traits::{
    Number,
    NumberFloat,
};
pub use crate::spatial::PointSet;

pub trait RawData: Sized {
    type Elem;
    fn data(&self) -> &[Self::Elem];
}

#[derive(Debug, Clone)]
pub struct MatrixBase<T, N = <T as RawData>::Elem>
where
    T: RawData<Elem = N>,
{
    data: T,
    dims: MatrixDims,
}

pub type Matrix<N> = MatrixBase<OwnedMatrixData<N>>;
pub type MatrixRef<'a, N> = MatrixBase<BorrowedMatrixData<'a, N>>;

impl<T, N> MatrixBase<T, N>
where
    T: RawData<Elem = N>,
{
    #[inline]
    pub fn to_matrixref(&self) -> MatrixRef<'_, N> {
        MatrixRef {
            data: self.data().into(),
            dims: self.dims(),
        }
    }
    #[inline]
    pub fn to_matrix(&self) -> Matrix<N>
    where
        N: Copy,
    {
        Matrix {
            data: self.data().to_vec().into(),
            dims: self.dims(),
        }
    }
    #[inline]
    pub fn data(&self) -> &[N] { self.data.data() }
    #[inline]
    pub fn dims(&self) -> MatrixDims { self.dims }
    #[inline]
    pub fn nrow(&self) -> NonZeroUsize { self.dims.rows }
    #[inline]
    pub fn ncol(&self) -> NonZeroUsize { self.dims.cols }
    #[inline]
    pub fn get<C>(&self, coord: C) -> Option<N>
    where
        C: Into<MatrixCoord>,
        N: Copy,
    {
        let coord = coord.into();
        self.dims.contains(coord).then(|| self[coord])
    }
    /// Returns an iterator on the row
    #[inline]
    pub fn row_iter(&self, row: usize) -> Option<RowIterator<'_, N>> { RowIterator::new(self, row) }
    /// Returns an iterator on the column
    #[inline]
    pub fn col_iter(&self, col: usize) -> Option<ColIterator<'_, N>> { ColIterator::new(self, col) }
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
        for mul in rhs.iter() {
            for pr in product.iter_mut() {
                *pr += *mul * self.data()[index];
                index += 1;
            }
        }
        Matrix::new(product, self.nrow())
    }
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
        for _ in 0..rhs.ncol().get() {
            let rhs_col = &rhs.data()[index..(index + rhs.nrow().get())];
            let res_col = self.mul_vec(rhs_col)?;
            product.extend_from_slice(res_col.data());
            index += rhs.nrow().get();
        }
        Matrix::new(product, self.nrow())
    }
}

impl<T, N, I> Index<I> for MatrixBase<T, N>
where
    T: RawData<Elem = N>,
    I: Into<MatrixCoord>,
{
    type Output = T::Elem;
    #[inline]
    fn index(&self, idx: I) -> &Self::Output {
        let index = idx.into().to_linear(self.dims).unwrap();
        &self.data()[index]
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
impl<'b, T, N> From<&'b MatrixBase<T, N>> for MatrixRef<'b, N>
where
    T: RawData<Elem = N>,
{
    #[inline]
    fn from(matrix: &'b MatrixBase<T, N>) -> Self { matrix.to_matrixref() }
}

impl<T, N> PointSet<N> for MatrixBase<T, N>
where
    T: RawData<Elem = N>,
{
    #[inline]
    fn size(&self) -> NonZeroUsize { self.dims.rows }
    #[inline]
    fn id_iter(&self) -> impl Iterator<Item = usize> { 0..self.dims.rows.get() }
    #[inline]
    fn dim(&self) -> NonZeroUsize { self.dims.cols }
    #[inline]
    fn exists(&self, id: usize) -> bool { id < self.dims.rows.get() }
    #[inline]
    fn coord(&self, id: usize, dim: usize) -> N
    where
        N: Copy,
    {
        let idx = id + dim * self.dims.rows.get();
        self.data()[idx]
    }
    #[inline]
    fn try_coord(&self, id: usize, dim: usize) -> Option<N>
    where
        N: Copy,
    {
        let idx = id + dim * self.dims.rows.get();
        self.data().get(idx).copied()
    }
    #[inline]
    fn sq_distance_between(&self, id_a: usize, id_b: usize) -> N
    where
        N: Number,
    {
        if id_a == id_b {
            return N::zero();
        }
        let mut idx = id_a.min(id_b);
        let idx_diff = id_a.abs_diff(id_b);
        let mut sum = N::zero();
        for _ in 0..self.ncol().get() {
            let diff = self.data()[idx] - self.data()[idx + idx_diff];
            sum += diff * diff;
            idx += self.nrow().get();
        }
        sum
    }
    #[inline]
    fn try_sq_distance_between(&self, id_a: usize, id_b: usize) -> Option<N>
    where
        N: Number,
    {
        (self.exists(id_a) && self.exists(id_b)).then(|| self.sq_distance_between(id_a, id_b))
    }
}

pub struct RowIterator<'a, N> {
    data: MatrixRef<'a, N>,
    coord: MatrixCoord,
}
impl<'a, N> RowIterator<'a, N> {
    #[inline]
    pub fn new<T>(matrix: &'a MatrixBase<T, N>, row: usize) -> Option<Self>
    where
        T: RawData<Elem = N>,
    {
        if !matrix.dims().contains_row(row) {
            return None;
        }
        let matrix = MatrixRef {
            data: matrix.data().into(),
            dims: matrix.dims(),
        };
        Self {
            data: matrix,
            coord: (row, 0).into(),
        }
        .into()
    }
}
impl<N> Iterator for RowIterator<'_, N>
where
    N: Copy,
{
    type Item = N;
    #[inline]
    fn next(&mut self) -> Option<N> {
        let val = self.data.get(self.coord);
        self.coord.col += 1;
        val
    }
}
impl<N> ExactSizeIterator for RowIterator<'_, N>
where
    N: Copy,
{
    #[inline]
    fn len(&self) -> usize { self.data.ncol().get() - self.coord.col }
}

pub struct ColIterator<'a, N> {
    data: MatrixRef<'a, N>,
    coord: MatrixCoord,
}
impl<'a, N> ColIterator<'a, N> {
    #[inline]
    pub fn new<T>(matrix: &'a MatrixBase<T, N>, col: usize) -> Option<Self>
    where
        T: RawData<Elem = N>,
    {
        if !matrix.dims().contains_col(col) {
            return None;
        }
        let matrix = MatrixRef {
            data: matrix.data().into(),
            dims: matrix.dims(),
        };
        Self {
            data: matrix,
            coord: (0, col).into(),
        }
        .into()
    }
}
impl<N> Iterator for ColIterator<'_, N>
where
    N: Copy,
{
    type Item = N;
    #[inline]
    fn next(&mut self) -> Option<N> {
        let val = self.data.get(self.coord);
        self.coord.row += 1;
        val
    }
}
impl<N> ExactSizeIterator for ColIterator<'_, N>
where
    N: Copy,
{
    #[inline]
    fn len(&self) -> usize { self.data.nrow().get() - self.coord.row }
}

use std::ops::IndexMut;

impl<N> Matrix<N> {
    #[inline]
    pub fn new(data: Vec<N>, rows: NonZeroUsize) -> Option<Self> {
        let dims = MatrixDims::from_row_count(data.len(), rows)?;
        let data = OwnedMatrixData::new(data);
        Some(Self { data, dims })
    }
    #[inline]
    pub fn from_value<D>(data: N, dims: D) -> Self
    where
        N: Copy,
        D: Into<MatrixDims>,
    {
        let dims: MatrixDims = dims.into();
        Self {
            data: OwnedMatrixData::new(vec![data; dims.len().get()]),
            dims: dims,
        }
    }
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

impl<'a, N> MatrixRef<'a, N> {
    #[inline]
    pub fn new(data: &'a [N], rows: NonZeroUsize) -> Option<Self> {
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
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        let index = idx.into().to_linear(self.dims()).unwrap();
        &mut self.data.data[index]
    }
}

#[derive(Debug, Clone)]
pub struct OwnedMatrixData<N> {
    data: Vec<N>,
}
impl<N> OwnedMatrixData<N> {
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

#[derive(Debug, Clone, Copy)]
pub struct BorrowedMatrixData<'a, N> {
    data: &'a [N],
}
impl<'a, N> BorrowedMatrixData<'a, N> {
    #[inline]
    pub fn new(data: &'a [N]) -> Self { Self { data } }
    #[inline]
    pub fn data(&self) -> &'a [N] { self.data }
}

impl<N> RawData for BorrowedMatrixData<'_, N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.data }
}
impl<'b, N> From<&'b OwnedMatrixData<N>> for BorrowedMatrixData<'b, N> {
    #[inline]
    fn from(data: &'b OwnedMatrixData<N>) -> Self { Self::new(data.data()) }
}
impl<'b, N> From<&'b [N]> for BorrowedMatrixData<'b, N> {
    #[inline]
    fn from(data: &'b [N]) -> Self { Self::new(data) }
}
