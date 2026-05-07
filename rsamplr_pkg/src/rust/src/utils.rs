use std::num::NonZeroUsize;

use num_traits::ToPrimitive;
use savvy::{
    OwnedIntegerSexp,
    Sexp,
    savvy_err,
};

pub fn to_usize<T>(v: T) -> savvy::Result<usize>
where
    T: ToPrimitive,
{
    v.to_usize()
        .ok_or_else(|| savvy_err!("value must be non-negative"))
}
pub fn to_nzusize<T>(v: T) -> savvy::Result<NonZeroUsize>
where
    T: ToPrimitive,
{
    v.to_usize()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| savvy_err!("value must be positive"))
}
pub fn to_i32<T>(v: T) -> savvy::Result<i32>
where
    T: ToPrimitive,
{
    v.to_i32()
        .ok_or_else(|| savvy_err!("cannot convert to i32"))
}

pub fn return_sample(sample: Vec<usize>) -> savvy::Result<Sexp> {
    let mut out = OwnedIntegerSexp::new(sample.len())?;

    for (i, &v) in sample.iter().enumerate() {
        out[i] = to_i32(v)? + 1;
    }

    Ok(Sexp::from(out))
}
