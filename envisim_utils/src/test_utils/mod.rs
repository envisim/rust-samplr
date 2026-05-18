mod data;

pub use data::*;

pub use crate::matrix::{
    MatrixBase,
    RawData,
};
pub use crate::number_traits::{
    Number,
    NumberFloat,
    NumberTest,
};

/// Helper for NonZeroUsize
pub fn nz(n: usize) -> std::num::NonZeroUsize { std::num::NonZeroUsize::new(n).unwrap() }

// pub const EPS: f64 = f64::TEST_EPS;

#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_delta {
    ($a:expr,$b:expr) => {{
        let a = $a;
        assert_delta!(a, $b, a.test_eps());
    }};
    ($a:expr,$b:expr,$d:expr) => {{
        let (a, b, eps) = ($a, $b, $d);
        assert!(a.approx_eq_eps(b, eps), "|{a} - {b}| >= {eps}");
    }};
}
pub use assert_delta;

#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_vec {
    ($v1:expr,$v2:expr) => {{
        let (v1, v2) = (&$v1, &$v2);
        assert_eq!(v1.len(), v2.len(), "vector dims do not match");
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            let eps = a.test_eps();
            assert!(a.approx_eq_eps(b, eps), "|{a} - {b}| >= {eps} (at {i})");
        }
    }};
    ($v1:expr,$v2:expr,$d:expr) => {{
        let (v1, v2, eps) = (&$v1, &$v2, $d);
        assert_eq!(v1.len(), v2.len(), "vector dims do not match");
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert!(a.approx_eq_eps(b, eps), "|{a} - {b}| >= {eps} (at {i})");
        }
    }};
}
pub use assert_vec;

#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_mat {
    ($m1:expr,$m2:expr) => {{
        let (m1, m2) = (&$m1, &$m2);
        let eps = m1[(0, 0)].test_eps();
        assert_mat!(m1, m2, eps);
    }};
    ($m1:expr,$m2:expr,$d:expr) => {{
        let (m1, m2, eps) = (&$m1, &$m2, $d);
        assert_eq!(m1.dims(), m2.dims(), "matrix dims do not match");
        for r in 0..m1.nrow().get() {
            for c in 0..m1.ncol().get() {
                let a = m1[(r, c)];
                let b = m2[(r, c)];
                assert!(a.approx_eq_eps(b, eps), "|{a} - {b}| >= {eps} (at {r},{c})");
            }
        }
    }};
}
pub use assert_mat;

// #[track_caller]
// #[inline]
// pub fn assert_fvec<N>(v1: &[N], v2: &[N])
// where
//     N: NumberTest,
// {
//     assert_eq!(v1.len(), v2.len(), "vector dims do not match");
//     for (&a, &b) in v1.iter().zip(v2.iter()) {
//         assert_delta!(a, b);
//     }
// }

// #[track_caller]
// #[inline]
// pub fn assert_fvec_eps<N>(v1: &[N], v2: &[N], eps: N)
// where
//     N: NumberTest,
// {
//     assert_eq!(v1.len(), v2.len(), "vector dims do not match");
//     for (&a, &b) in v1.iter().zip(v2.iter()) {
//         assert_delta!(a, b, eps);
//     }
// }

// #[track_caller]
// #[inline]
// pub fn assert_fmat<T1, T2, N>(m1: &MatrixBase<T1, N>, m2: &MatrixBase<T2, N>)
// where
//     T1: RawData<Elem = N>,
//     T2: RawData<Elem = N>,
//     N: NumberTest,
// {
//     assert_eq!(m1.dims(), m2.dims(), "matrix dims do not match");
//     for (&a, &b) in m1.data().data().iter().zip(m2.data().data().iter()) {
//         assert_delta!(a, b);
//     }
// }

// #[track_caller]
// #[inline]
// pub fn assert_fmat_eps<T1, T2, N>(m1: &MatrixBase<T1, N>, m2: &MatrixBase<T2, N>, eps: N)
// where
//     T1: RawData<Elem = N>,
//     T2: RawData<Elem = N>,
//     N: NumberTest,
// {
//     assert_eq!(m1.dims(), m2.dims(), "matrix dims do not match");
//     for (&a, &b) in m1.data().data().iter().zip(m2.data().data().iter()) {
//         assert_delta!(a, b, eps);
//     }
// }
