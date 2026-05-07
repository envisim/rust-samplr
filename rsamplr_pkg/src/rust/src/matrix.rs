use std::num::NonZeroUsize;

pub use envisim_utils::matrix::Dimensions;
use envisim_utils::matrix::MatrixRef;
use savvy::{
    RealSexp,
    savvy_err,
};

pub fn get_nrow(mat: &RealSexp) -> savvy::Result<NonZeroUsize> {
    let rows: usize = mat.get_dim().ok_or(savvy_err!("object is not matrix"))?[0]
        .try_into()
        .map_err(|_| savvy_err!("dimension must be positive"))?;
    NonZeroUsize::new(rows).ok_or(savvy_err!("dimension must be positive"))
}

pub fn to_matrix(mat: &[f64], nrow: NonZeroUsize) -> MatrixRef<'_, f64> {
    MatrixRef::new(mat, nrow).unwrap()
}
