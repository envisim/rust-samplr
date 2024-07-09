pub fn usize_to_f64(v: usize) -> f64 {
    match u32::try_from(v) {
        Ok(v) => return f64::from(v),
        _ => panic!("usize ({}) too large to be converted to f64", v),
    };
}
