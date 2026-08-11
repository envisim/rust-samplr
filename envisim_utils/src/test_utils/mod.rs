// Copyright (C) 2025 Wilmer Prentius.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Test utility functions, enabled by feature `test-utils`.

mod data;

pub use data::*;

pub use crate::matrix::MatrixBase;
pub use crate::utils::{
    Epsilon,
    Number,
    NumberFloat,
    NumberInt,
    SliceView,
};

/// Helper for NonZeroUsize
pub fn nz(n: usize) -> std::num::NonZeroUsize { std::num::NonZeroUsize::new(n).unwrap() }

/// Macro for asserting float-like-equalities
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_delta {
    ($a:expr,$b:expr) => {{
        let a = $a;
        assert_delta!(a, $b, a.default_epsilon());
    }};
    ($a:expr,$b:expr,$d:expr) => {{
        let (a, b, eps) = ($a, $b, $d);
        assert!(eps.difference_is_zero(a, b), "|{a} - {b}| > {eps}");
    }};
}
pub use assert_delta;

/// Macro for asserting float-like-equalities for array-likes
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_vec {
    ($v1:expr,$v2:expr) => {{
        let (v1, v2) = (&$v1, &$v2);
        assert_eq!(v1.len(), v2.len(), "vector dims do not match");
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            let eps = a.default_epsilon();
            assert!(eps.difference_is_zero(a, b), "|{a} - {b}| > {eps} (at {i})");
        }
    }};
    ($v1:expr,$v2:expr,$d:expr) => {{
        let (v1, v2, eps) = (&$v1, &$v2, $d);
        assert_eq!(v1.len(), v2.len(), "vector dims do not match");
        for (i, (&a, &b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert!(eps.difference_is_zero(a, b), "|{a} - {b}| > {eps} (at {i})");
        }
    }};
}
pub use assert_vec;

/// Macro for asserting float-like-equalities for matrices
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! assert_mat {
    ($m1:expr,$m2:expr) => {{
        let (m1, m2) = (&$m1, &$m2);
        let eps = m1[(0, 0)].default_epsilon();
        assert_mat!(m1, m2, eps);
    }};
    ($m1:expr,$m2:expr,$d:expr) => {{
        let (m1, m2, eps) = (&$m1, &$m2, $d);
        assert_eq!(m1.dims(), m2.dims(), "matrix dims do not match");
        for r in 0..m1.nrow().get() {
            for c in 0..m1.ncol().get() {
                let a = m1[(r, c)];
                let b = m2[(r, c)];
                assert!(
                    eps.difference_is_zero(a, b),
                    "|{a} - {b}| > {eps} (at {r},{c})"
                );
            }
        }
    }};
}
pub use assert_mat;
