pub use envisim_utils::random::RandomNumberGenerator;

extern "C" {
    fn GetRNGstate();
    fn PutRNGstate();
    fn unif_rand() -> f64;
    // fn norm_rand() -> f64;
    // fn exp_rand() -> f64;
}

pub struct RRng();

impl RRng {
    pub fn new() -> Self {
        unsafe {
            GetRNGstate();
        }
        RRng()
    }
}

impl Drop for RRng {
    fn drop(&mut self) {
        unsafe {
            PutRNGstate();
        }
    }
}

impl RandomNumberGenerator for RRng {
    #[inline]
    fn rf64(&mut self) -> f64 { unsafe { unif_rand() } }
    #[inline]
    fn ru32(&mut self) -> u32 { self.ri32() as u32 }
    #[inline]
    fn ru32_to(&mut self, b: u32) -> u32 {
        loop {
            let u = self.ru32();
            let m = u32::MAX - (u32::MAX % b);
            if u < m {
                return u % b;
            }
        }
    }
    #[inline]
    fn ri32(&mut self) -> i32 {
        let f = self.rf64() * 2.0 - 1.0;
        (f * (i32::MAX as f64)).floor() as i32
    }
    #[inline]
    fn ri32_in(&mut self, a: i32, b: i32) -> Option<i32> {
        self.rf64_in(a.into(), b.into()).map(|v| v.floor() as i32)
    }
    #[inline]
    fn ru64(&mut self) -> u64 {
        let high = self.ru32() as u64;
        let low = self.ru32() as u64;
        (high << 32) | low
    }
    #[inline]
    fn ru64_to(&mut self, b: u64) -> u64 {
        loop {
            let u = self.ru64();
            let m = u64::MAX - (u64::MAX % b);
            if u < m {
                return u % b;
            }
        }
    }
    #[inline]
    fn ri64(&mut self) -> i64 { self.ru64() as i64 }
    #[inline]
    fn ri64_in(&mut self, a: i64, b: i64) -> Option<i64> {
        if b < a {
            return None;
        } else if a == b {
            return Some(a);
        }

        let d = (b as u64).wrapping_sub(a as u64);
        // d can be 0...u32::MAX...i64::MAX / infty

        if d <= (u32::MAX as u64) {
            return Some(self.ru32_to(d as u32) as i64 + a);
        } else {
            loop {
                let u = self.ru64() & 0x7FFF_FFFF_FFFF_FFFF;
                let m = u64::MAX - (u64::MAX) % d;
                if u < m {
                    return Some((u % d) as i64 + a);
                }
            }
        }
    }
}
