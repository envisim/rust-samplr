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

//! Energy distance utils

use std::num::NonZeroUsize;

use envisim_utils::spatial::PointSet;
use num_traits::ToPrimitive;

/// Energy distance engine (or n-energy distance)
#[must_use]
#[derive(Clone, Debug)]
pub struct EnergyDistance<P> {
    /// Distribution matrix
    matrix: P,
    /// Geometric potentials
    phi: Vec<f64>,
    /// Population total spread
    u_spread: f64,
    /// Sample size
    sample_size: NonZeroUsize,
}

impl<P> EnergyDistance<P> {
    /// Constructs a new energy distance engine
    #[inline]
    pub fn new(matrix: P, sample_size: NonZeroUsize) -> Self
    where
        P: PointSet<N = f64>,
    {
        let size = matrix.size().get();
        let u_size = size.to_f64().expect("matrix size to convert to f64");
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

        u_spread *= sample_size
            .get()
            .to_f64()
            .expect("sample_size to convert to f64")
            / u_size;

        Self {
            matrix,
            phi,
            u_spread,
            sample_size,
        }
    }
    /// Returns the geometric potential of `id`
    #[must_use]
    #[inline]
    fn geometric_potential(&self, id: usize) -> f64 { self.phi[id] }
    /// Returns the energy between `id1` and `id2`
    #[must_use]
    #[inline]
    fn s_energy_between(&self, id1: usize, id2: usize) -> f64
    where
        P: PointSet<N = f64>,
    {
        self.matrix.sq_distance_between(id1, id2).sqrt()
    }
    /// Returns the relative distance between `unit` -- `a` and `unit` -- `b`
    #[must_use]
    #[inline]
    pub fn relative_distance(&self, unit: usize, a: usize, b: usize) -> f64
    where
        P: PointSet<N = f64>,
    {
        self.matrix.sq_distance_between(unit, a).sqrt()
            - self.matrix.sq_distance_between(unit, b).sqrt()
    }
    /// The total energy of all samples
    #[must_use]
    #[inline]
    pub fn total<I>(&self, sample_iter: I) -> f64
    where
        P: PointSet<N = f64>,
        I: Iterator<Item = usize> + Clone,
    {
        let mut s_spread: f64 = 0.0;
        let mut inter_spread: f64 = 0.0;

        // let mut outer = sample_iter.clone();
        let mut outer = sample_iter;

        while let Some(id1) = outer.next() {
            inter_spread += self.phi[id1];

            // Inner starts one after outer
            let inner = outer.clone();
            for id2 in inner {
                s_spread += self.s_energy_between(id1, id2);
            }
        }

        inter_spread *= 2.0;
        s_spread *= 2.0
            / self
                .sample_size
                .get()
                .to_f64()
                .expect("sample_size to convert to f64");
        inter_spread - s_spread - self.u_spread
    }
    /// Returns the inter-energy delta
    #[must_use]
    #[inline]
    fn i_delta(&self, add: usize, rem: usize) -> Option<f64> {
        if add == rem {
            return None;
        }
        let inter_spread = (self.geometric_potential(add) - self.geometric_potential(rem)) * 2.0;
        Some(inter_spread)
    }
    /// Returns the sample-energy delta
    #[must_use]
    #[inline]
    fn s_delta<I>(&self, sample: I, add: usize, rem: usize) -> Option<f64>
    where
        P: PointSet<N = f64>,
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

        s_spread *= 2.0
            / self
                .sample_size
                .get()
                .to_f64()
                .expect("sample_size to convert to f64");
        Some(s_spread)
    }
    /// Returns the energy delta
    #[must_use]
    #[inline]
    pub fn delta<I>(&self, sample: I, add: usize, rem: usize) -> Option<f64>
    where
        P: PointSet<N = f64>,
        I: Iterator<Item = usize> + Clone,
    {
        let inter_spread = self.i_delta(add, rem)?;
        let s_spread = self.s_delta(sample, add, rem)?;
        Some(inter_spread + s_spread)
    }
}
