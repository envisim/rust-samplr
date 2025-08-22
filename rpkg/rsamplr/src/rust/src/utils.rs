use std::num::NonZeroUsize;

use savvy::{savvy_err, OwnedIntegerSexp, Sexp};

pub fn i32_to_usize(v: i32) -> savvy::Result<usize> {
    v.try_into()
        .map_err(|_| savvy_err!("value must be non-negative"))
}

pub fn i32_to_nonzerousize(v: i32) -> savvy::Result<NonZeroUsize> {
    let uv = i32_to_usize(v)?;
    NonZeroUsize::new(uv).ok_or(savvy_err!("value must be positive"))
}

pub fn usize_to_i32(v: usize) -> savvy::Result<i32> {
    v.try_into()
        .map_err(|_| savvy_err!("cannot convert from usize to i32"))
}

pub fn return_sample(sample: Vec<usize>) -> savvy::Result<Sexp> {
    let mut out = OwnedIntegerSexp::new(sample.len())?;

    for (i, &v) in sample.iter().enumerate() {
        out[i] = usize_to_i32(v)? + 1;
    }

    Ok(Sexp::from(out))
}
