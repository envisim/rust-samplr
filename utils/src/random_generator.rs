pub trait RandomGenerator {
    fn rf64(&mut self) -> f64;
    #[inline]
    fn rf64_scaled(&mut self, scale: f64) -> f64 {
        self.rf64() * scale
    }
    #[inline]
    fn one_of_f64(&mut self, v0: f64, v1: f64) -> bool {
        self.rf64_scaled(v0 + v1) < v1
    }
    #[inline]
    fn ri64(&mut self, scale: i64) -> i64 {
        if scale == 0 || scale == 1 {
            return 0;
        }

        self.rf64_scaled(scale as f64) as i64
    }
    #[allow(clippy::cast_possible_wrap)]
    #[inline]
    fn rusize(&mut self, scale: usize) -> usize {
        self.ri64(scale as i64) as usize
    }
    #[inline]
    fn rslice<'t, T>(&mut self, slice: &'t [T]) -> Option<&'t T> {
        let k = self.rusize(slice.len());
        slice.get(k)
    }
}

pub trait RandomList: RandomGenerator {
    fn in_list(&mut self, _idx: usize) -> f64 {
        self.rf64()
    }
}

pub struct Constant {
    value: f64,
}
impl Constant {
    pub const fn new(value: f64) -> Self {
        Constant { value }
    }
}
impl RandomGenerator for Constant {
    fn rf64(&mut self) -> f64 {
        self.value
    }
}
impl RandomList for Constant {}

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
