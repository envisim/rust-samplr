pub struct Matrix<'a> {
    dt: &'a [f64],
    n: usize,
    p: usize,
}

impl<'a> Matrix<'a> {
    #[inline]
    pub fn new(dt: &[f64], n: usize) -> Matrix {
        let p = dt.len() / n;

        assert!(p > 0);
        assert!(dt.len() % p == 0);

        Matrix { dt, p, n }
    }

    #[inline]
    pub fn get(&self, id: usize, k: usize) -> f64 {
        self.dt[k * self.n + id]
    }

    #[inline]
    pub unsafe fn get_unsafe(&self, id: usize, k: usize) -> f64 {
        *self.dt.get_unchecked(k * self.n + id)
    }

    #[inline]
    pub unsafe fn get_distance(&self, id: usize, unit: &[f64]) -> f64 {
        let mut k: usize = 0;
        let mut index: usize = id;
        let mut distance: f64 = 0.0;
        while k < self.p {
            let temp: f64 = unit.get_unchecked(k) - self.dt.get_unchecked(index);
            distance += temp * temp;
            k += 1;
            index += self.n;
        }

        return distance;
    }

    #[inline]
    pub fn into_unit_iter(&self, id: usize) -> UnitIterator {
        UnitIterator {
            matrix: self,
            index: id,
        }
    }

    #[inline]
    pub fn into_var_iter(&self, k: usize) -> VarIterator {
        VarIterator {
            matrix: self,
            index: k * self.n,
        }
    }

    #[inline]
    pub fn ncol(&self) -> usize {
        self.p
    }

    #[inline]
    pub fn nrow(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (self.n, self.p)
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.dt.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_sizes() {
        let data1 = Matrix::new(&[2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2], 9);
        assert!(data1.nrow() == 9);
        assert!(data1.ncol() == 1);

        let data2 = Matrix::new(
            &[
                0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
                0.66079779, 0.62911404, 0.06178627, //
                0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185,
                0.9919061, 0.3800352, 0.7774452,
            ],
            10,
        );

        assert!(data2.nrow() == 10);
        assert!(data2.ncol() == 2);
    }
}

pub struct UnitIterator<'a> {
    matrix: &'a Matrix<'a>,
    index: usize,
}

pub struct VarIterator<'a> {
    matrix: &'a Matrix<'a>,
    index: usize,
}

impl<'a> Iterator for UnitIterator<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.matrix.dt.get(self.index).cloned();
        self.index += self.matrix.n;
        result
    }
}

impl<'a> Iterator for VarIterator<'a> {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.matrix.dt.get(self.index).cloned();
        self.index += 1;
        result
    }
}
