use envisim_samplr_utils::matrix::RefMatrix;
use envisim_samplr_utils::random_generator::{Constant, RandomGenerator};
use rand::{rngs::SmallRng, Rng, SeedableRng};

#[allow(dead_code)]
macro_rules! assert_delta {
    ($a:expr,$b:expr,$d:expr) => {
        assert!(($a - $b).abs() < $d, "|{} - {}| >= {}", $a, $b, $d);
    };
}

pub(crate) use assert_delta;

#[allow(dead_code)]
pub const fn gen_rand() -> (Constant, Constant) {
    (gen_rand00(), gen_rand99())
}
#[allow(dead_code)]
pub const fn gen_rand00() -> Constant {
    Constant::new(0.0)
}
#[allow(dead_code)]
pub const fn gen_rand99() -> Constant {
    Constant::new(0.999)
}
#[allow(dead_code)]
pub const EPS: f64 = 1e-12;

pub struct TestRandom {
    rand: SmallRng,
}
impl TestRandom {
    pub fn new(seed: u64) -> Self {
        TestRandom {
            rand: SmallRng::seed_from_u64(seed),
        }
    }
}
impl RandomGenerator for TestRandom {
    #[inline]
    fn rf64(&mut self) -> f64 {
        self.rand.gen()
    }
}

// DISTS:
//            0         1         2         3         4         5         6         7         8         9
// 0  0.0000000 0.1105994 0.5708486 0.6669274 0.5674677 0.6968799 0.8503216 0.8797396 0.4031203 0.6066972
// 1  0.1105994 0.0000000 0.5485142 0.5748579 0.6172820 0.6165129 0.7877581 0.8649434 0.3277917 0.6762960
// 2  0.5708486 0.5485142 0.0000000 0.4519099 0.3802988 0.3765864 0.3730786 0.3173138 0.3121004 0.5190046
// 3  0.6669274 0.5748579 0.4519099 0.0000000 0.8049673 0.1140190 0.3355026 0.6562282 0.2791234 0.9333525
// 4  0.5674677 0.6172820 0.3802988 0.8049673 0.0000000 0.7479726 0.7448264 0.5100001 0.5784869 0.1401022
// 5  0.6968799 0.6165129 0.3765864 0.1140190 0.7479726 0.0000000 0.2247373 0.5483524 0.2938608 0.8821355
// 6  0.8503216 0.7877581 0.3730786 0.3355026 0.7448264 0.2247373 0.0000000 0.3947406 0.4621054 0.8849137
// 7  0.8797396 0.8649434 0.3173138 0.6562282 0.5100001 0.5483524 0.3947406 0.0000000 0.6126907 0.6362455
// 8  0.4031203 0.3277917 0.3121004 0.2791234 0.5784869 0.2938608 0.4621054 0.6126907 0.0000000 0.6926727
// 9  0.6066972 0.6762960 0.5190046 0.9333525 0.1401022 0.8821355 0.8849137 0.6362455 0.6926727 0.0000000

// NEIGHBOURS (row's closest neighbours):
// 0:  0 1 8 4 2 9 3 5 6 7
// 1:  1 0 8 2 3 5 4 9 6 7
// 2:  2 8 7 6 5 4 3 9 1 0
// 3:  3 5 8 6 2 1 7 0 4 9
// 4:  4 9 2 7 0 8 1 6 5 3
// 5:  5 3 6 8 2 7 1 0 4 9
// 6:  6 5 3 2 7 8 4 1 0 9
// 7:  7 2 6 4 5 8 9 3 1 0
// 8:  8 3 5 2 1 0 6 4 7 9
// 9:  9 4 2 0 7 1 8 5 6 3

#[allow(dead_code)]
pub const DATA_10_2: [f64; 20] = [
    0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527, 0.66079779,
    0.62911404, 0.06178627, //
    0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
    0.3800352, 0.7774452,
];

#[allow(dead_code)]
pub fn data_10_2<'a>() -> (RefMatrix<'a>, [f64; 10]) {
    (RefMatrix::new(&DATA_10_2, 10), [0.2f64; 10])
}

#[allow(dead_code)]
pub const DATA_20_2: [f64; 40] = [
    0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527, 0.66079779,
    0.62911404, 0.06178627, 1.26550866, 1.37212390, 1.57285336, 1.90820779, 1.20168193, 1.89838968,
    1.94467527, 1.66079779, 1.62911404, 1.06178627, //
    0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
    0.3800352, 0.7774452, 0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992,
    0.7176185, 0.9919061, 0.3800352, 0.7774452,
];

#[allow(dead_code)]
pub fn data_20_2<'a>() -> (RefMatrix<'a>, [f64; 20]) {
    (RefMatrix::new(&DATA_20_2, 20), [0.2f64; 20])
}
