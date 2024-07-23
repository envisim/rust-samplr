use std::iter::{Skip, StepBy};
use std::ops::{Index, IndexMut};
use std::slice::Iter;

type MatrixIndex = (usize, usize);

pub trait OperateMatrix: Index<MatrixIndex, Output = f64> {
    fn data(&self) -> &[f64];
    fn nrow(&self) -> usize;
    fn ncol(&self) -> usize;
    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.nrow(), self.ncol())
    }

    #[inline]
    fn into_row_iter(&self, row: usize) -> MatrixIterator {
        assert!(row < self.nrow());
        MatrixIterator {
            iter: self.data().iter().skip(row).step_by(self.nrow()),
            dim: self.dim(),
            step: self.nrow(),
            count: 0,
        }
    }

    #[inline]
    fn into_col_iter<'a>(&self, col: usize) -> MatrixIterator {
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

    #[inline]
    unsafe fn get_unchecked(&self, (row, col): MatrixIndex) -> &f64 {
        self.data().get_unchecked(col * self.nrow() + row)
    }

    #[inline]
    fn distance_to_row(&self, row: usize, unit: &[f64]) -> f64 {
        assert!(unit.len() == self.ncol());
        assert!(row < self.nrow());

        self.data()
            .iter()
            .skip(row)
            .step_by(self.nrow())
            .zip(unit.iter())
            .fold(0.0, |acc, (a, b)| acc + (a - b).powi(2))
    }

    /// Performes the calculation of self * multiplicand, where self
    /// is a matrix A, and multiplicand is a vector
    #[inline]
    fn prod_vec(&self, multiplicand: &[f64]) -> Vec<f64> {
        assert!(multiplicand.len() == self.ncol());
        let data = self.data();
        let mut prod = vec![0.0; self.nrow()];
        let mut index: usize = 0;

        for j in 0..self.ncol() {
            for i in 0..self.nrow() {
                prod[i] += multiplicand[j] * data[index];
                index += 1;
            }
        }

        prod
    }

    /// Performes the calculation of matrices self * mat
    fn mult<M>(&self, mat: &M) -> Matrix
    where
        M: OperateMatrix,
    {
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
    fn len(&self) -> usize {
        if self.step == 1 {
            self.dim.0 - self.count
        } else {
            self.dim.1 - self.count
        }
    }
}

pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

pub struct RefMatrix<'a> {
    data: &'a [f64],
    rows: usize,
    cols: usize,
}

impl Matrix {
    #[inline]
    pub fn new(data: &[f64], rows: usize) -> Self {
        assert!(rows > 0);
        assert!(data.len() % rows == 0);
        let cols = data.len() / rows;
        Self {
            data: data.to_vec(),
            rows: rows,
            cols: cols,
        }
    }

    pub fn new_fill(data: f64, (rows, cols): MatrixIndex) -> Self {
        assert!(rows > 0);
        assert!(cols > 0);
        Self {
            data: vec![data; rows * cols],
            rows: rows,
            cols: cols,
        }
    }

    pub unsafe fn resize(&mut self, rows: usize, cols: usize) {
        assert!(rows > 0 && cols > 0);
        let new_size = rows * cols;
        self.data.resize(new_size, 0.0);
        self.rows = rows;
        self.cols = cols;
    }

    pub fn reduced_row_echelon_form(&mut self) {
        let mut lead: usize = 0;

        for row in 0..self.rows {
            if self.cols <= lead {
                return;
            }

            let mut i: usize = row;

            while unsafe { *self.get_unchecked((i, lead)) == 0.0 } {
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
                    self.data.swap(index_i, index_row);
                    index_i += self.rows;
                    index_row += self.rows;
                }
            }

            // Divide ROW by lead, assuming all is 0 before lead
            let mut index_row = row + lead * self.rows;
            let lead_value = unsafe { *self.data.get_unchecked(index_row) };
            if lead_value != 1.0 {
                unsafe {
                    *self.data.get_unchecked_mut(index_row) = 1.0;
                }
                index_row += self.rows;

                for _ in (lead + 1)..self.cols {
                    unsafe {
                        *self.data.get_unchecked_mut(index_row) /= lead_value;
                    }
                    index_row += self.rows;
                }
            }

            // Remove ROW from all other rows
            for j in 0..self.rows {
                if j == row {
                    continue;
                }

                let mut index_j = j + lead * self.rows;
                let lead_multiplicator = unsafe { *self.data.get_unchecked(index_j) };
                if lead_multiplicator == 0.0 {
                    continue;
                }
                unsafe {
                    *self.data.get_unchecked_mut(index_j) = 0.0;
                }
                index_j += self.rows;
                let mut index_row = row + (lead + 1) * self.rows;

                for _ in (lead + 1)..self.cols {
                    unsafe {
                        *self.data.get_unchecked_mut(index_j) -=
                            *self.data.get_unchecked(index_row) * lead_multiplicator;
                    }
                    index_j += self.rows;
                    index_row += self.rows;
                }
            }

            lead += 1;
        }
    }
}

impl<'a> RefMatrix<'a> {
    #[inline]
    pub fn new(data: &[f64], rows: usize) -> RefMatrix {
        assert!(rows > 0);
        assert!(data.len() % rows == 0);
        let cols = data.len() / rows;
        RefMatrix { data, rows, cols }
    }

    #[inline]
    pub fn from_matrix(mat: &Matrix) -> RefMatrix {
        assert!(mat.nrow() > 0);
        assert!(mat.ncol() > 0);
        RefMatrix {
            data: mat.data(),
            rows: mat.nrow(),
            cols: mat.ncol(),
        }
    }
}

impl Index<MatrixIndex> for Matrix {
    type Output = f64;

    #[inline]
    fn index(&self, (row, col): MatrixIndex) -> &f64 {
        assert!(col < self.cols && row < self.rows);
        unsafe { self.data.get_unchecked(col * self.rows + row) }
    }
}

impl IndexMut<MatrixIndex> for Matrix {
    #[inline]
    fn index_mut(&mut self, (row, col): MatrixIndex) -> &mut f64 {
        assert!(col < self.cols && row < self.rows);
        unsafe { self.data.get_unchecked_mut(col * self.rows + row) }
    }
}

impl<'a> Index<MatrixIndex> for RefMatrix<'a> {
    type Output = f64;

    #[inline]
    fn index(&self, (row, col): MatrixIndex) -> &f64 {
        assert!(col < self.cols && row < self.rows);
        unsafe { self.data.get_unchecked(col * self.rows + row) }
    }
}

impl OperateMatrix for Matrix {
    #[inline]
    fn data(&self) -> &[f64] {
        &self.data
    }
    #[inline]
    fn nrow(&self) -> usize {
        self.rows
    }
    #[inline]
    fn ncol(&self) -> usize {
        self.cols
    }
}

impl<'a> OperateMatrix for RefMatrix<'a> {
    #[inline]
    fn data(&self) -> &[f64] {
        self.data
    }
    #[inline]
    fn nrow(&self) -> usize {
        self.rows
    }
    #[inline]
    fn ncol(&self) -> usize {
        self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_delta, DATA_10_2, EPS};

    #[test]
    fn matrix_sizes() {
        let data1 = Matrix::new(&[2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2], 9);
        assert!(data1.nrow() == 9);
        assert!(data1.ncol() == 1);

        let data2 = Matrix::new(&DATA_10_2, 10);
        assert!(data2.nrow() == 10);
        assert!(data2.ncol() == 2);
    }

    #[test]
    fn iterator() {
        let data1 = Matrix::new(&[0.0, 1.0, 2.0, 0.1, 1.1, 2.1, 0.2, 1.2, 2.2], 3);
        let mut iter = data1.into_row_iter(1);
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(&1.0));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(&1.1));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some(&1.2));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);

        iter = data1.into_col_iter(1);
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(&0.1));
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some(&1.1));
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some(&2.1));
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn rref() {
        let mut data1 = Matrix::new(
            &[
                0.81, 0.46, 0.40, //
                0.54, 0.70, 0.08, //
                0.39, 0.42, 0.87, //
                0.64, 0.70, 0.32, //
            ],
            3,
        );
        data1.reduced_row_echelon_form();
        assert_delta!(data1[(0, 0)], 1.0, EPS); // COL1
        assert_delta!(data1[(1, 0)], 0.0, EPS);
        assert_delta!(data1[(2, 0)], 0.0, EPS);
        assert_delta!(data1[(0, 1)], 0.0, EPS); // COL2
        assert_delta!(data1[(1, 1)], 1.0, EPS);
        assert_delta!(data1[(2, 1)], 0.0, EPS);
        assert_delta!(data1[(0, 2)], 0.0, EPS); // COL3
        assert_delta!(data1[(1, 2)], 0.0, EPS);
        assert_delta!(data1[(2, 2)], 1.0, EPS);
        assert_delta!(data1[(0, 3)], 0.188953701217875, EPS); // COL4
        assert_delta!(data1[(1, 3)], 0.748566128914163, EPS);
        assert_delta!(data1[(2, 3)], 0.212107159999675, EPS);
    }

    #[test]
    fn mult() {
        let data1 = Matrix::new(
            &[
                1.00, 0.52, 0.96, 0.91, 0.12, 0.70, 0.10, 0.38, 0.05, 0.27, 0.52, 0.25,
            ],
            3,
        );
        let vec1 = vec![0.75f64, 0.27, 0.24, 0.3];
        let res1 = data1.prod_vec(&vec1);
        let facit1 = vec![1.1007, 0.6696, 0.9960];
        assert_delta!(res1[0], facit1[0], EPS);
        assert_delta!(res1[1], facit1[1], EPS);
        assert_delta!(res1[2], facit1[2], EPS);

        let data2 = Matrix::new(&[0.45, 0.84, 0.86, 0.09, 0.60, 0.13, 0.18, 0.69], 4);
        let res = data1.mult(&data2);
        let facit = vec![1.3247, 0.7084, 1.0855, 0.9226, 0.7548, 0.8485];
        assert_eq!(res.data().len(), facit.len());
        assert_delta!(res.data()[0], facit[0], EPS);
        assert_delta!(res.data()[1], facit[1], EPS);
        assert_delta!(res.data()[2], facit[2], EPS);
        assert_delta!(res.data()[3], facit[3], EPS);
        assert_delta!(res.data()[4], facit[4], EPS);
        assert_delta!(res.data()[5], facit[5], EPS);
    }
}
