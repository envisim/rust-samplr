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

use std::num::NonZeroUsize;

use envisim_utils::spatial::PointSet;
use envisim_utils::utils::usize_to_f64;

/// Energy distance cointainer (or n-energy distance)
#[derive(Clone, Debug)]
pub struct EnergyDistance<P> {
    matrix: P,
    phi: Vec<f64>,
    u_spread: f64,
    sample_size: NonZeroUsize,
}

impl<P> EnergyDistance<P> {
    pub fn new(matrix: P, sample_size: NonZeroUsize) -> Self
    where
        P: PointSet<f64>,
    {
        let size = matrix.size().get();
        let u_size = usize_to_f64(size);
        let mut phi = vec![0.0; size];
        let mut u_spread = 0.0;

        for id1 in matrix.id_iter() {
            for id2 in matrix.id_iter().skip(id1 + 1) {
                let dist = matrix.sq_distance_between(id1, id2).sqrt();
                phi[id1] += dist;
                phi[id2] += dist;
            }
            phi[id1] /= u_size;
            u_spread += phi[id1];
        }

        u_spread *= usize_to_f64(sample_size.get()) / u_size;

        Self {
            matrix,
            phi,
            u_spread,
            sample_size,
        }
    }
    fn geometric_potential(&self, id: usize) -> f64 { self.phi[id] }
    fn s_energy_between(&self, id1: usize, id2: usize) -> f64
    where
        P: PointSet<f64>,
    {
        self.matrix.sq_distance_between(id1, id2).sqrt()
    }
    pub fn relative_distance(&self, unit: usize, a: usize, b: usize) -> f64
    where
        P: PointSet<f64>,
    {
        self.matrix.sq_distance_between(unit, a).sqrt()
            - self.matrix.sq_distance_between(unit, b).sqrt()
    }
    pub fn total<I>(&self, sample_iter: I) -> f64
    where
        P: PointSet<f64>,
        I: Iterator<Item = usize> + Clone,
    {
        let mut s_spread: f64 = 0.0;
        let mut inter_spread: f64 = 0.0;

        let mut outer = sample_iter.clone();

        while let Some(id1) = outer.next() {
            inter_spread += self.phi[id1];

            // Inner starts one after outer
            let inner = outer.clone();
            for id2 in inner {
                s_spread += self.s_energy_between(id1, id2);
            }
        }

        inter_spread *= 2.0;
        s_spread *= 2.0 / usize_to_f64(self.sample_size.get());
        inter_spread - s_spread - self.u_spread
    }
    fn i_delta(&self, add: usize, rem: usize) -> Option<f64> {
        if add == rem {
            return None;
        }
        let inter_spread = (self.geometric_potential(add) - self.geometric_potential(rem)) * 2.0;
        Some(inter_spread)
    }
    fn s_delta<I>(&self, sample: I, add: usize, rem: usize) -> Option<f64>
    where
        P: PointSet<f64>,
        I: Iterator<Item = usize> + Clone,
    {
        if add == rem {
            return None;
        }

        let mut s_spread = 0.0;
        for id in sample {
            if id == add {
                // Would lead to two add
                return None;
            } else if id == rem {
                continue;
            }

            s_spread += self.relative_distance(id, rem, add);
        }

        s_spread *= 2.0 / usize_to_f64(self.sample_size.get());
        Some(s_spread)
    }
    pub fn delta<I>(&self, sample: I, add: usize, rem: usize) -> Option<f64>
    where
        P: PointSet<f64>,
        I: Iterator<Item = usize> + Clone,
    {
        let inter_spread = self.i_delta(add, rem)?;
        let s_spread = self.s_delta(sample, add, rem)?;
        Some(inter_spread + s_spread)
    }
}
