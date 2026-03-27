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

use envisim_utils::random::RandomNumberGenerator;

/// Random permutation of usize [0,...,len] vector
pub fn shuffled_indices<R: RandomNumberGenerator>(rng: &mut R, len: usize) -> Vec<usize> {
    assert!(len > 0, "len must be positive");

    let mut order: Vec<usize> = Vec::with_capacity(len);
    order.push(0);

    for i in 1..len {
        let j = rng.rusize_to(i + 1);
        order.push(i);
        order.swap(i, j);
    }

    order
}
