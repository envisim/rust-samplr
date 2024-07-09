pub trait GenerateRandom {
    fn random_float(&self) -> f64;
    #[inline]
    fn random_float_scale(&self, scale: f64) -> f64 {
        self.random_float() * scale
    }
    #[inline]
    fn random_int_scale(&self, scale: i64) -> i64 {
        if scale == 0 || scale == 1 {
            return 0;
        }

        self.random_float_scale(scale as f64) as i64
    }
    #[inline]
    fn random_usize_scale(&self, scale: usize) -> usize {
        self.random_int_scale(scale as i64) as usize
    }
    #[inline]
    fn random_get<'t, T>(&self, slice: &'t [T]) -> Option<&'t T> {
        let k = self.random_usize_scale(slice.len());
        slice.get(k)
    }
    #[inline]
    fn random_float_of(&self, _k: usize) -> f64 {
        self.random_float()
    }
}

pub struct StaticRandom {
    value: f64,
}
impl StaticRandom {
    pub const fn new(value: f64) -> Self {
        StaticRandom { value: value }
    }
}
impl GenerateRandom for StaticRandom {
    fn random_float(&self) -> f64 {
        self.value
    }
}

// Needed for R maybe
// #[inline]
// pub fn uniform_std(rand: &RandomGenerator) -> f64 {
//     let mut u: f64;
//     loop {
//         u = rand();
//         if u >= 0.0 && u < 1.0 {
//             return u;
//         }
//     }
// }
