#[inline]
pub fn usize_to_f64(v: usize) -> f64 {
    match u32::try_from(v) {
        Ok(v) => f64::from(v),
        _ => panic!("usize ({}) too large to be converted to f64", v),
    }
}

#[inline]
pub fn sum(vec: &[f64]) -> f64 {
    vec.iter().fold(0.0, |acc, x| acc + x)
}

#[inline]
pub fn mean(vec: &[f64]) -> f64 {
    sum(vec) / usize_to_f64(vec.len())
}

#[inline]
pub fn variance(vec: &[f64]) -> f64 {
    if vec.len() == 1 {
        return f64::NAN;
    }

    let mean = mean(vec);
    vec.iter().fold(0.0, |acc, x| acc + (x - mean).powi(2)) / usize_to_f64(vec.len() - 1)
}

#[inline]
pub fn standard_deviance(vec: &[f64]) -> f64 {
    variance(vec).sqrt()
}

#[inline]
pub fn random_element<'t, R, T>(rng: &mut R, slice: &'t [T]) -> Option<&'t T>
where
    R: rand::Rng + ?Sized,
{
    if slice.is_empty() {
        return None;
    }

    let k: usize = rng.gen_range(0..slice.len());
    Some(&slice[k])
}

#[inline]
pub fn random_one_of_f64<R>(rng: &mut R, v0: f64, v1: f64) -> bool
where
    R: rand::Rng + ?Sized,
{
    rng.gen_range(0.0..(v0 + v1)) < v1
}
