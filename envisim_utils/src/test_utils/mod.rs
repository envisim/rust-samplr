mod data;

pub use data::*;

/// Helper for NonZeroUsize
pub fn nz(n: usize) -> std::num::NonZeroUsize { std::num::NonZeroUsize::new(n).unwrap() }

pub const EPS: f64 = 1e-12;

#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_delta {
    ($a:expr,$b:expr,$d:expr) => {
        assert!(($a - $b).abs() < $d, "|{} - {}| >= {}", $a, $b, $d);
    };

    ($a:expr,$b:expr) => {
        assert!(($a - $b).abs() < EPS, "|{} - {}| >= {}", $a, $b, EPS);
    };
}
pub use assert_delta;

pub fn assert_fvec(v1: &[f64], v2: &[f64]) {
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_delta!(a, b);
    }
}

pub fn assert_fvec_eps(v1: &[f64], v2: &[f64], d: f64) {
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_delta!(a, b, d);
    }
}
