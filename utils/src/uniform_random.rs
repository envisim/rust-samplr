pub type RandomGenerator = fn() -> f64;

#[inline]
pub fn uniform_std(rand: &RandomGenerator) -> f64 {
    let mut u: f64;
    loop {
        u = rand();
        if u >= 0.0 && u < 1.0 {
            return u;
        }
    }
}

#[inline]
pub fn uniform_max(rand: &RandomGenerator, max: f64) -> f64 {
    uniform_std(rand) * max
}

#[inline]
pub fn discrete_uniform(rand: &RandomGenerator, max: i64) -> i64 {
    if max == 0 || max == 1 {
        0
    } else {
        uniform_max(rand, max as f64) as i64
    }
}

#[inline]
pub fn discrete_uniform_u(rand: &RandomGenerator, max: usize) -> usize {
    assert!(max > 0);

    if max == 1 {
        0
    } else {
        uniform_max(rand, max as f64) as usize
    }
}

pub trait DrawRandom<T> {
    fn draw(&self, rand: &RandomGenerator) -> T;
}

impl<T: Copy> DrawRandom<T> for Vec<T> {
    #[inline]
    fn draw(&self, rand: &RandomGenerator) -> T {
        let k = discrete_uniform_u(rand, self.len());
        self[k]
    }
}

impl<T: Copy> DrawRandom<T> for [T] {
    #[inline]
    fn draw(&self, rand: &RandomGenerator) -> T {
        let k = discrete_uniform_u(rand, self.len());
        self[k]
    }
}

impl<T: Copy> DrawRandom<T> for &[T] {
    #[inline]
    fn draw(&self, rand: &RandomGenerator) -> T {
        let k = discrete_uniform_u(rand, self.len());
        self[k]
    }
}
