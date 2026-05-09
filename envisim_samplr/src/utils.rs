// Copyright (C) 2026 Wilmer Prentius.
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

//! Utility functions for sampling algorithms

use std::num::NonZeroUsize;

use envisim_utils::random::RandomNumberGenerator;

/// Random permutation of usize [0,...,len] vector
#[inline]
pub fn shuffled_indices<R>(rng: &mut R, len: NonZeroUsize) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    let mut order: Vec<usize> = Vec::with_capacity(len.get());
    order.push(0);

    for i in 1..len.get() {
        let j = rng.rusize_to(i + 1);
        order.push(i);
        order.swap(i, j);
    }

    order
}

/// The basic poisson sampling algorithm
#[inline]
pub fn poisson_internal<R>(rng: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: RandomNumberGenerator,
{
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| rng.rbern(p).and_then(|b| b.then_some(i)))
        .collect()
}

/// Returns the greatest common divisor of `a` and `b`
#[inline]
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    if a == 0 {
        return b;
    } else if b == 0 || a == b {
        return a;
    }

    // Count common factors of 2
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    b >>= b.trailing_zeros();

    while a != b {
        if a > b {
            a -= b;
            a >>= a.trailing_zeros();
        } else {
            b -= a;
            b >>= b.trailing_zeros();
        }
    }

    a << shift
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 5), 5); // One number is zero
        assert_eq!(gcd(7, 0), 7); // Other number is zero
        assert_eq!(gcd(1, 100), 1); // GCD is 1 (coprime)
        assert_eq!(gcd(48, 18), 6); // Standard case
        assert_eq!(gcd(100, 35), 5); // Standard case
        assert_eq!(gcd(17, 17), 17); // Both numbers equal
        assert_eq!(gcd(1024, 512), 512); // One divides the other (powers of 2)
        assert_eq!(gcd(97, 89), 1); // Two primes (coprime)
        assert_eq!(gcd(462, 1071), 21); // Larger numbers with non-trivial GCD
        assert_eq!(gcd(123456, 789012), 12); // Large numbers
    }
}
