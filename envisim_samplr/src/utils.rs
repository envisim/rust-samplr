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

use envisim_utils::random::Rand;

/// Random permutation of usize [0,...,len] vector
#[inline]
pub fn shuffled_indices<R>(rng: &mut R, len: NonZeroUsize) -> Vec<usize>
where
    R: Rand<usize>,
{
    let mut order: Vec<usize> = Vec::with_capacity(len.get());
    order.push(0);

    for i in 1..len.get() {
        let j = rng.rand_to(i + 1);
        order.push(i);
        order.swap(i, j);
    }

    order
}

/// The basic poisson sampling algorithm
#[inline]
pub fn poisson_internal<R, I>(rng: &mut R, probabilities: I) -> Vec<usize>
where
    R: Rand<f64>,
    I: Iterator<Item = f64>,
{
    probabilities
        .enumerate()
        .filter_map(|(i, p)| (rng.rand() < p).then_some(i))
        .collect()
}

#[cfg(test)]
mod test {
    // use super::*;
}
