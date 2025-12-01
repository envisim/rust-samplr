use envisim_utils::Matrix;
use savvy::{savvy_err, RealSexp};

pub fn get_nrow(mat: &RealSexp) -> savvy::Result<usize> {
    mat.get_dim().ok_or(savvy_err!("object is not matrix"))?[0]
        .try_into()
        .map_err(|_| savvy_err!("dimension must be positive"))
}

pub fn to_matrix(mat: &[f64], nrow: usize) -> Matrix {
    Matrix::new(mat, nrow).unwrap()
}
