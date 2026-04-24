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

use neighbour::{
    Neighbour,
    WeightedNeighbour,
};

use super::{
    PointAccess,
    Tree,
};

pub mod neighbour {
    #[derive(Copy, Clone, Debug)]
    pub struct Neighbour {
        id: usize,
        distance: f64,
    }
    impl Neighbour {
        #[inline]
        pub fn id(&self) -> usize { self.id }
        #[inline]
        pub fn distance(&self) -> f64 { self.distance }
        #[inline]
        pub fn new(id: usize, distance: f64) -> Self { Self { id, distance } }
    }
    impl PartialEq for Neighbour {
        fn eq(&self, other: &Self) -> bool { self.distance.total_cmp(&other.distance).is_eq() }
    }
    impl Eq for Neighbour {}
    impl Ord for Neighbour {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.distance.total_cmp(&other.distance)
        }
    }
    impl PartialOrd for Neighbour {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct WeightedNeighbour {
        neighbour: Neighbour,
        weight: f64,
    }

    impl WeightedNeighbour {
        #[inline]
        pub fn id(&self) -> usize { self.neighbour.id }
        #[inline]
        pub fn distance(&self) -> f64 { self.neighbour.distance }
        #[inline]
        pub fn weight(&self) -> f64 { self.weight }
        #[inline]
        pub fn new(id: usize, distance: f64, weight: f64) -> Self {
            let neighbour = Neighbour::new(id, distance);
            Self { neighbour, weight }
        }
    }
    impl PartialEq for WeightedNeighbour {
        fn eq(&self, other: &Self) -> bool { self.cmp(other).is_eq() }
    }
    impl Eq for WeightedNeighbour {}
    impl Ord for WeightedNeighbour {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.distance()
                .total_cmp(&other.distance())
                .then(self.weight().total_cmp(&other.weight()))
        }
    }
    impl PartialOrd for WeightedNeighbour {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
    }

    pub trait ToNeighbourIds {
        fn to_neighbour_ids(&self) -> Box<[usize]>;
    }
    impl ToNeighbourIds for &[Neighbour] {
        #[inline]
        fn to_neighbour_ids(&self) -> Box<[usize]> {
            self.iter()
                .map(|n| n.id())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }
    }
    impl ToNeighbourIds for &[WeightedNeighbour] {
        #[inline]
        fn to_neighbour_ids(&self) -> Box<[usize]> {
            self.iter()
                .map(|n| n.id())
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }
    }
}

pub trait TreeSearcher {
    fn point(&self) -> &[f64];
    fn is_satisfied(&self, distance: f64) -> bool;
    fn visit_leaf<T>(&mut self, data: &T, leaf_units: &[usize])
    where
        T: PointAccess;
}

#[derive(Clone, Debug)]
pub struct SearchPoint {
    point: Box<[f64]>,
    unit: Option<usize>,
}
impl SearchPoint {
    #[inline]
    pub fn point(&self) -> &[f64] { &self.point }
    #[inline]
    pub fn unit(&self) -> Option<usize> { self.unit }
    #[inline]
    pub fn is_unit(&self, other: usize) -> bool { self.unit.is_some_and(|u| u == other) }
    #[inline]
    pub fn new(dim: NonZeroUsize) -> Self {
        Self {
            point: vec![0.0; dim.get()].into_boxed_slice(),
            unit: None,
        }
    }
    #[inline]
    pub fn from_unit<T>(data: &T, unit: usize) -> Option<Self>
    where
        T: PointAccess,
    {
        Self {
            point: data.to_boxed_slice(unit)?,
            unit: unit.into(),
        }
        .into()
    }
    #[inline]
    pub fn from_slice(point: &[f64]) -> Option<Self> {
        if point.is_empty() {
            return None;
        }
        Self {
            point: point.into(),
            unit: None,
        }
        .into()
    }
    #[inline]
    pub fn set_from_unit<T>(&mut self, data: &T, unit: usize) -> Option<()>
    where
        T: PointAccess,
    {
        self.point = data.to_boxed_slice(unit)?;
        self.unit = Some(unit);
        Some(())
    }
    #[inline]
    pub fn set_from_slice(&mut self, point: &[f64]) -> Option<()> {
        (self.point.len() == point.len()).then(|| {
            self.point = point.into();
            self.unit = None;
        })
    }
}

/// Searching the nearest neighbour of a unit.
#[derive(Clone, Debug)]
pub struct NearestNeighbourSearcher {
    point: SearchPoint,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`]
    neighbours: Vec<Neighbour>,
}
impl NearestNeighbourSearcher {
    #[inline]
    pub fn new<T>(data: &T) -> Self
    where
        T: PointAccess,
    {
        Self {
            point: SearchPoint::new(data.dim()),
            neighbours: Vec::with_capacity(6),
        }
    }
    #[inline]
    pub fn from_unit<T>(data: &T, unit: usize) -> Option<Self>
    where
        T: PointAccess,
    {
        Self {
            point: SearchPoint::from_unit(data, unit)?,
            neighbours: Vec::with_capacity(6),
        }
        .into()
    }
    #[inline]
    pub fn from_slice(point: &[f64]) -> Option<Self> {
        Self {
            point: SearchPoint::from_slice(point)?,
            neighbours: Vec::with_capacity(6),
        }
        .into()
    }
    #[inline]
    pub fn reset_from_unit<T>(&mut self, data: &T, unit: usize) -> Option<&mut Self>
    where
        T: PointAccess,
    {
        self.point.set_from_unit(data, unit)?;
        self.neighbours.clear();
        self.into()
    }
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[f64]) -> Option<&mut Self> {
        self.point.set_from_slice(point)?;
        self.neighbours.clear();
        self.into()
    }
    /// Finds the nearest neighbour of the search point. In cases of ties, all nearest neighbours
    /// are added.
    #[inline]
    pub fn search<T>(&mut self, tree: &Tree<T>) -> Option<()>
    where
        T: PointAccess,
    {
        self.neighbours.clear();
        tree.iterate_leafs_by(self)
    }
    #[inline]
    pub fn neighbours(&self) -> &[Neighbour] { &self.neighbours }
    #[inline]
    fn max_distance(&self) -> f64 {
        self.neighbours
            .last()
            .map_or(f64::INFINITY, |n| n.distance())
    }
}
impl TreeSearcher for NearestNeighbourSearcher {
    #[inline]
    fn point(&self) -> &[f64] { self.point.point() }
    #[inline]
    fn is_satisfied(&self, distance: f64) -> bool {
        // Satisfied only if enough units AND a potential unit is not further away
        !self.neighbours.is_empty() && self.max_distance() < distance.powi(2)
    }
    fn visit_leaf<T>(&mut self, data: &T, leaf_units: &[usize])
    where
        T: PointAccess,
    {
        let mut current_max = self.max_distance();
        for &id in leaf_units.iter() {
            if Some(id) == self.point.unit {
                continue;
            }

            let distance = data.sq_distance(id, self.point());
            if distance < current_max {
                self.neighbours.clear();
                self.neighbours.push(Neighbour::new(id, distance));
                current_max = distance;
            } else if distance == current_max {
                self.neighbours.push(Neighbour::new(id, distance));
            }
        }
    }
}

/// Searching the k nearest neighbour of a unit.
#[derive(Clone, Debug)]
pub struct KNearestNeighbourSearcher {
    point: SearchPoint,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`]
    neighbours: Vec<Neighbour>,
    nominal_size: NonZeroUsize,
}
impl KNearestNeighbourSearcher {
    #[inline]
    pub fn new<T>(k: NonZeroUsize, data: &T) -> Self
    where
        T: PointAccess,
    {
        Self {
            point: SearchPoint::new(data.dim()),
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        }
    }
    #[inline]
    pub fn from_unit<T>(k: NonZeroUsize, data: &T, unit: usize) -> Option<Self>
    where
        T: PointAccess,
    {
        Self {
            point: SearchPoint::from_unit(data, unit)?,
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        }
        .into()
    }
    #[inline]
    pub fn from_slice(k: NonZeroUsize, point: &[f64]) -> Option<Self> {
        Self {
            point: SearchPoint::from_slice(point)?,
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        }
        .into()
    }
    #[inline]
    pub fn set_nominal_size(&mut self, k: NonZeroUsize) -> &mut Self {
        self.nominal_size = k;
        self
    }
    #[inline]
    pub fn reset_from_unit<T>(&mut self, data: &T, unit: usize) -> Option<&mut Self>
    where
        T: PointAccess,
    {
        self.point.set_from_unit(data, unit)?;
        self.neighbours.clear();
        self.into()
    }
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[f64]) -> Option<&mut Self> {
        self.point.set_from_slice(point)?;
        self.neighbours.clear();
        self.into()
    }
    /// Finds the k nearest neighbour of the search point. In cases of ties, more than k neighbours
    /// might be added.
    #[inline]
    pub fn search<T>(&mut self, tree: &Tree<T>) -> Option<()>
    where
        T: PointAccess,
    {
        self.neighbours.clear();
        tree.iterate_leafs_by(self)
    }
    #[inline]
    pub fn neighbours(&self) -> &[Neighbour] { &self.neighbours }
    #[inline]
    fn max_distance(&self) -> f64 {
        self.neighbours
            .last()
            .map_or(f64::INFINITY, |n| n.distance())
    }
}
impl TreeSearcher for KNearestNeighbourSearcher {
    #[inline]
    fn point(&self) -> &[f64] { self.point.point() }
    #[inline]
    fn is_satisfied(&self, distance: f64) -> bool {
        // Satisfied only if enough units AND a potential unit is not further away
        self.neighbours.len() >= self.nominal_size.get() && self.max_distance() < distance.powi(2)
    }
    fn visit_leaf<T>(&mut self, data: &T, leaf_units: &[usize])
    where
        T: PointAccess,
    {
        let original_len = self.neighbours.len();

        // Default to 0.0 b/c trick below.
        // self.neighbours is assumed to be sorted by distance.
        let mut current_max = self.neighbours.last().map_or(0.0, |n| n.distance());
        for &id in leaf_units.iter() {
            if Some(id) == self.point.unit {
                continue;
            }

            // The slightly weird logic below guarantees that current_max is always the value of the
            // currently largest added unit. We will add all units that are smaller than this to the
            // potential units
            let distance = data.sq_distance(id, self.point());
            if distance <= current_max {
                self.neighbours.push(Neighbour::new(id, distance));
            } else if self.neighbours.len() < self.nominal_size.get() {
                self.neighbours.push(Neighbour::new(id, distance));
                // Remember: we are only here because the unit is BOTH further away from current max
                // AND we haven't filled up the number of units yet
                current_max = distance;
            }
        }

        let new_len = self.neighbours.len();
        if new_len == original_len {
            // Since store size has not changed, we have not added any units -- no need to sort
            return;
        }

        self.neighbours.sort_unstable();
        // Set current max to the max value according to the sort, at the kth unit

        if new_len <= self.nominal_size.get() {
            // Store contains no additional units
            return;
        }

        // Find units to cut off. Every unit w/ equal distance as the kth unit should be kept
        let last_idx = self.nominal_size.get() - 1; // Guaranteed > 0
        current_max = self.neighbours[last_idx].distance();
        // Get the partition point of all units equal to this (lower units should be sorted before)
        let partition_point = self.neighbours[self.nominal_size.get()..]
            .partition_point(|x| x.distance() <= current_max)
            + self.nominal_size.get();
        // Truncate the vector
        self.neighbours.truncate(partition_point);
    }
}

/// A weighted nearest neighbour search of a unit, where weights are assumed to be probabilities
#[derive(Clone, Debug)]
pub struct WeightedSearcher {
    point: SearchPoint,
    point_weight: f64,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`], where lower weights are
    /// sorted before higher weights in case of ties.
    neighbours: Vec<WeightedNeighbour>,
    total_weight: f64,
}
impl WeightedSearcher {
    #[inline]
    pub fn new<T>(data: &T) -> Self
    where
        T: PointAccess,
    {
        Self {
            point: SearchPoint::new(data.dim()),
            point_weight: 0.5,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        }
    }
    #[inline]
    pub fn from_unit<T>(data: &T, unit: usize, weight: f64) -> Option<Self>
    where
        T: PointAccess,
    {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        Self {
            point: SearchPoint::from_unit(data, unit)?,
            point_weight: weight,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        }
        .into()
    }
    #[inline]
    pub fn from_slice(point: &[f64], weight: f64) -> Option<Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        Self {
            point: SearchPoint::from_slice(point)?,
            point_weight: weight,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        }
        .into()
    }
    #[inline]
    pub fn reset_from_unit<T>(&mut self, data: &T, unit: usize, weight: f64) -> Option<&mut Self>
    where
        T: PointAccess,
    {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        self.point.set_from_unit(data, unit)?;
        self.point_weight = weight;
        self.neighbours.clear();
        self.total_weight = 0.0;
        self.into()
    }
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[f64], weight: f64) -> Option<&mut Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        self.point.set_from_slice(point)?;
        self.point_weight = weight;
        self.neighbours.clear();
        self.total_weight = 0.0;
        self.into()
    }
    /// Finds the nearest neighbour of the search point, such that the nearest neighbours add up to
    /// at least 1.0 total weight. In cases of ties, more neighbours might be added.
    #[inline]
    pub fn search<T, W>(&mut self, tree: &Tree<T>, weights: &W) -> Option<()>
    where
        T: PointAccess,
        W: WeightCollection,
    {
        if !(0.0 < self.point_weight && self.point_weight < 1.0) {
            return None;
        }
        self.neighbours.clear();
        self.total_weight = 0.0;
        let mut searcher = WeightedSearcherWrapper::new(self, weights);
        tree.iterate_leafs_by(&mut searcher)
    }
    #[inline]
    pub fn neighbours(&self) -> &[WeightedNeighbour] { &self.neighbours }
    #[inline]
    pub fn total_weight(&self) -> f64 { self.total_weight }
    #[inline]
    pub fn point_weight(&self) -> f64 { self.point_weight }
    #[inline]
    fn max_distance(&self) -> f64 {
        self.neighbours
            .last()
            .map_or(f64::INFINITY, |n| n.distance())
    }
    #[inline]
    fn calculate_weight(&self, other: f64) -> f64 {
        if !(0.0 < other && other < 1.0) {
            0.0
        } else if self.point_weight + other <= 1.0 {
            other / (1.0 - self.point_weight)
        } else {
            (1.0 - other) / self.point_weight
        }
    }
}

pub trait WeightCollection {
    fn try_get_weight(&self, id: usize) -> Option<f64>;
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self.try_get_weight(id).expect("id to exist") }
}
impl WeightCollection for &[f64] {
    #[inline]
    fn try_get_weight(&self, id: usize) -> Option<f64> { self.get(id).copied() }
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self[id] }
}

#[derive(Debug)]
struct WeightedSearcherWrapper<'a, W>
where
    W: WeightCollection,
{
    searcher: &'a mut WeightedSearcher,
    weights: &'a W,
}
impl<'a, W> WeightedSearcherWrapper<'a, W>
where
    W: WeightCollection,
{
    fn new(searcher: &'a mut WeightedSearcher, weights: &'a W) -> Self {
        Self { searcher, weights }
    }
}
impl<'a, W> TreeSearcher for WeightedSearcherWrapper<'a, W>
where
    W: WeightCollection,
{
    #[inline]
    fn point(&self) -> &[f64] { self.searcher.point.point() }
    #[inline]
    fn is_satisfied(&self, distance: f64) -> bool {
        self.searcher.total_weight >= 1.0 && self.searcher.max_distance() < distance.powi(2)
    }
    fn visit_leaf<T>(&mut self, data: &T, leaf_units: &[usize])
    where
        T: PointAccess,
    {
        let original_len = self.searcher.neighbours.len();

        // Default to 0.0 b/c trick below
        let mut current_max = self
            .searcher
            .neighbours
            .last()
            .map_or(0.0, |n| n.distance());
        for &id in leaf_units.iter() {
            if Some(id) == self.searcher.point.unit {
                continue;
            }

            // The slightly weird logic below guarantees that current_max is always the value of the
            // currently largest added unit. We will add all units that are smaller than this to the
            // potential units
            let distance = data.sq_distance(id, self.point());
            if distance <= current_max {
                let weight = self.searcher.calculate_weight(self.weights.get_weight(id));
                self.searcher
                    .neighbours
                    .push(WeightedNeighbour::new(id, distance, weight));
            } else if self.searcher.total_weight < 1.0 {
                let weight = self.searcher.calculate_weight(self.weights.get_weight(id));
                self.searcher
                    .neighbours
                    .push(WeightedNeighbour::new(id, distance, weight));
                // Remember: we are only here because the unit is BOTH further away from current max
                // AND we haven't filled up the number of units yet
                current_max = distance;
            }
        }

        let new_len = self.searcher.neighbours.len();
        if new_len == original_len {
            // Since store size has not changed, we have not added any units -- no need to sort
            return;
        }

        // Sorts by distance, then by weight, so lower weight is before higher weights
        self.searcher.neighbours.sort_unstable();

        // TRUNCATE
        let mut w_sum = 0.0;
        // Gives the units that are guaranteed to be included b/c weight is not full
        let last_idx = self
            .searcher
            .neighbours
            .iter()
            .position(|n| {
                w_sum += n.weight();
                w_sum >= 1.0
            })
            .unwrap_or(new_len);
        // Move one passed, so all good units are in the first partition

        let mut partition_point = last_idx + 1;
        if new_len <= partition_point {
            // If we have not filled up the total weight, we do not need to truncate
            self.searcher.total_weight = w_sum;
            return;
        }

        // Find units to cut off. Every unit w/ equal distance as the kth unit should be kept
        // Set current max to the max value according to the current partition point
        current_max = self.searcher.neighbours[last_idx].distance();
        // Now we need to find units on the same distance as the last unit in safe_idx
        // If below is None, then all remaining units are on the same distance as last_idx, and
        // should all be kept.
        if let Some(sub_partition_point) = self.searcher.neighbours[partition_point..]
            .iter()
            .position(|n| {
                if n.distance() > current_max {
                    return true;
                }
                w_sum += n.weight();
                false
            })
        {
            partition_point += sub_partition_point;
            // Truncate the vector
            self.searcher.neighbours.truncate(partition_point);
        }

        self.searcher.total_weight = w_sum;
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::Tree;
    use crate::kd_tree::searcher::{
        KNearestNeighbourSearcher,
        NearestNeighbourSearcher,
        SearchPoint,
        WeightCollection,
        WeightedSearcher,
    };
    use crate::matrix::Matrix;
    use crate::sampling_options::SpreadingOptions;

    fn test_builder<'a>(data: Matrix<'a>) -> SpreadingOptions<'a> {
        SpreadingOptions::new(data).unwrap()
    }

    fn mat_from(data: Vec<f64>, rows: usize) -> Matrix<'static> {
        Matrix::from_vec(data, NonZeroUsize::new(rows).unwrap()).unwrap()
    }

    // 5 points in 2D along the x-axis
    // (0,0), (1,0), (2,0), (3,0), (10,0)
    fn line_5_points() -> Matrix<'static> {
        mat_from(vec![0.0, 1.0, 2.0, 3.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5)
    }

    // --- SearchPoint ---

    #[test]
    fn search_point_new_zero_fills() {
        let sp = SearchPoint::new(NonZeroUsize::new(3).unwrap());
        assert_eq!(sp.point(), &[0.0, 0.0, 0.0]);
        assert!(sp.unit().is_none());
    }

    #[test]
    fn search_point_from_unit_copies_row() {
        let m = line_5_points();
        let sp = SearchPoint::from_unit(&m, 2).unwrap();
        assert_eq!(sp.point(), &[2.0, 0.0]);
        assert_eq!(sp.unit(), Some(2));
        assert!(sp.is_unit(2));
        assert!(!sp.is_unit(3));
    }

    #[test]
    fn search_point_from_unit_invalid_id_returns_none() {
        let m = line_5_points();
        assert!(SearchPoint::from_unit(&m, 99).is_none());
    }

    #[test]
    fn search_point_from_slice_empty_returns_none() {
        assert!(SearchPoint::from_slice(&[]).is_none());
    }

    #[test]
    fn search_point_set_from_slice_enforces_dim() {
        let m = line_5_points();
        let mut sp = SearchPoint::from_unit(&m, 0).unwrap();
        // Wrong length
        assert!(sp.set_from_slice(&[1.0]).is_none());
        // Right length
        assert!(sp.set_from_slice(&[5.0, 5.0]).is_some());
        assert_eq!(sp.point(), &[5.0, 5.0]);
        // unit is now cleared
        assert!(sp.unit().is_none());
    }

    // --- Neighbour comparisons ---

    #[test]
    fn neighbour_ord_is_by_distance() {
        use crate::kd_tree::searcher::neighbour::Neighbour;
        let a = Neighbour::new(0, 1.0);
        let b = Neighbour::new(1, 2.0);
        let c = Neighbour::new(2, 1.0);
        assert!(a < b);
        assert!(c < b);
        assert_eq!(a.cmp(&c), std::cmp::Ordering::Equal);
    }

    // --- NearestNeighbourSearcher ---

    #[test]
    fn nearest_neighbour_finds_closest() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        // Query from unit 2 (at x=2). Closest are x=1 and x=3, both at distance 1.
        let mut searcher = NearestNeighbourSearcher::from_unit(tree.data(), 2).unwrap();
        searcher.search(&tree).unwrap();

        let neighbours = searcher.neighbours();
        assert_eq!(neighbours.len(), 2, "two tied nearest neighbours");
        for n in neighbours {
            assert_eq!(n.distance(), 1.0);
            assert!(n.id() == 1 || n.id() == 3);
        }
    }

    #[test]
    fn nearest_neighbour_excludes_self() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let mut searcher = NearestNeighbourSearcher::from_unit(tree.data(), 0).unwrap();
        searcher.search(&tree).unwrap();
        for n in searcher.neighbours() {
            assert_ne!(n.id(), 0);
        }
    }

    #[test]
    fn nearest_neighbour_from_arbitrary_slice() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        // Query at (1.5, 0). Closest are id 1 (x=1) and id 2 (x=2), both distance 0.25.
        let mut searcher = NearestNeighbourSearcher::from_slice(&[1.5, 0.0]).unwrap();
        searcher.search(&tree).unwrap();
        let ns = searcher.neighbours();
        assert_eq!(ns.len(), 2);
        for n in ns {
            assert!((n.distance() - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn nearest_neighbour_reset_clears_state() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let mut searcher = NearestNeighbourSearcher::from_unit(tree.data(), 2).unwrap();
        searcher.search(&tree).unwrap();
        assert!(!searcher.neighbours().is_empty());

        searcher.reset_from_unit(tree.data(), 0).unwrap();
        assert!(searcher.neighbours().is_empty());
    }

    #[test]
    fn nearest_neighbour_from_slice_wrong_dim_returns_none() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let mut searcher = NearestNeighbourSearcher::from_slice(&[0.0]).unwrap();
        // 1D query on a 2D tree -> iterate_leafs_by returns None
        assert!(searcher.search(&tree).is_none());
    }

    // --- KNearestNeighbourSearcher ---

    #[test]
    fn k_nearest_returns_k_units() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let k = NonZeroUsize::new(3).unwrap();
        let mut s = KNearestNeighbourSearcher::from_unit(k, tree.data(), 0).unwrap();
        s.search(&tree).unwrap();

        let ns = s.neighbours();
        assert_eq!(ns.len(), 3);
        // Expected: ids 1, 2, 3 at distances 1, 4, 9
        let mut ids: Vec<usize> = ns.iter().map(|n| n.id()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3]);

        // Sorted ascending by distance
        for pair in ns.windows(2) {
            assert!(pair[0].distance() <= pair[1].distance());
        }
    }

    #[test]
    fn k_nearest_keeps_ties_beyond_k() {
        // 4 points all at distance 1 from query
        // Query at (0,0); points at (1,0), (-1,0), (0,1), (0,-1)
        let m = mat_from(
            vec![
                1.0, -1.0, 0.0, 0.0, // dim 0
                0.0, 0.0, 1.0, -1.0, // dim 1
            ],
            4,
        );
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..4).collect();
        let tree = Tree::new(&builder, &mut units);

        let k = NonZeroUsize::new(2).unwrap();
        let mut s = KNearestNeighbourSearcher::from_slice(k, &[0.0, 0.0]).unwrap();
        s.search(&tree).unwrap();

        // All four units are tied at distance 1.0 from (0,0), so ties rule
        // keeps all of them.
        assert_eq!(s.neighbours().len(), 4);
        for n in s.neighbours() {
            assert!((n.distance() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn k_nearest_with_k_larger_than_population_returns_all() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let k = NonZeroUsize::new(100).unwrap();
        let mut s = KNearestNeighbourSearcher::from_unit(k, tree.data(), 0).unwrap();
        s.search(&tree).unwrap();
        assert_eq!(s.neighbours().len(), 4); // all except self
    }

    #[test]
    fn k_nearest_excludes_self() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let k = NonZeroUsize::new(2).unwrap();
        let mut s = KNearestNeighbourSearcher::from_unit(k, tree.data(), 3).unwrap();
        s.search(&tree).unwrap();
        for n in s.neighbours() {
            assert_ne!(n.id(), 3);
        }
    }

    #[test]
    fn k_nearest_set_nominal_size_takes_effect_after_reset() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let k = NonZeroUsize::new(1).unwrap();
        let mut s = KNearestNeighbourSearcher::from_unit(k, tree.data(), 0).unwrap();
        s.set_nominal_size(NonZeroUsize::new(3).unwrap());
        s.reset_from_unit(tree.data(), 0).unwrap();
        s.search(&tree).unwrap();
        assert_eq!(s.neighbours().len(), 3);
    }

    // --- WeightCollection for &[f64] ---

    #[test]
    fn weight_collection_slice_get_weight() {
        let w: &[f64] = &[0.1, 0.2, 0.3];
        assert_eq!(w.get_weight(0), 0.1);
        assert_eq!(w.try_get_weight(2), Some(0.3));
        assert_eq!(w.try_get_weight(99), None);
    }

    // --- WeightedSearcher ---

    #[test]
    fn weighted_searcher_rejects_out_of_range_weight() {
        let m = line_5_points();
        assert!(WeightedSearcher::from_unit(&m, 0, 0.0).is_none());
        assert!(WeightedSearcher::from_unit(&m, 0, 1.0).is_none());
        assert!(WeightedSearcher::from_unit(&m, 0, -0.1).is_none());
        assert!(WeightedSearcher::from_unit(&m, 0, 1.5).is_none());
        assert!(WeightedSearcher::from_unit(&m, 0, 0.5).is_some());
    }

    #[test]
    fn weighted_searcher_accumulates_weight_to_one() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        // All weights 0.5 ; point_weight = 0.5 ; so each other's weight is
        // 0.5 / (1 - 0.5) = 1.0. So we stop after accumulating 1 neighbour.
        let weights: Vec<f64> = vec![0.5; 5];
        let w_slice: &[f64] = &weights;
        let mut s = WeightedSearcher::from_unit(tree.data(), 2, 0.5).unwrap();
        s.search(&tree, &w_slice).unwrap();

        // total_weight should be >= 1.0
        assert!(s.total_weight() >= 1.0);
        // Neighbours should have at least one entry
        assert!(!s.neighbours().is_empty());
    }

    #[test]
    fn weighted_searcher_point_weight_accessor() {
        let m = line_5_points();
        let s = WeightedSearcher::from_unit(&m, 0, 0.25).unwrap();
        assert_eq!(s.point_weight(), 0.25);
    }

    #[test]
    fn weighted_searcher_reset_clears_state() {
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        let weights: Vec<f64> = vec![0.5; 5];
        let w_slice: &[f64] = &weights;
        let mut s = WeightedSearcher::from_unit(tree.data(), 2, 0.5).unwrap();
        s.search(&tree, &w_slice).unwrap();
        assert!(!s.neighbours().is_empty());

        s.reset_from_unit(tree.data(), 0, 0.4).unwrap();
        assert!(s.neighbours().is_empty());
        assert_eq!(s.total_weight(), 0.0);
        assert_eq!(s.point_weight(), 0.4);
    }

    #[test]
    fn weighted_searcher_reset_rejects_bad_weight() {
        let m = line_5_points();
        let mut s = WeightedSearcher::from_unit(&m, 0, 0.5).unwrap();
        assert!(s.reset_from_unit(&m, 1, 0.0).is_none());
        assert!(s.reset_from_unit(&m, 1, 1.0).is_none());
        assert!(s.reset_from_slice(&[0.0, 0.0], 1.5).is_none());
    }

    #[test]
    fn weighted_searcher_respects_tight_weights() {
        // With very small weights, many units are needed to accumulate w_sum >= 1
        let m = line_5_points();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..5).collect();
        let tree = Tree::new(&builder, &mut units);

        // point_weight 0.01; other weights 0.2 -> each contributes
        // 0.2 / (1 - 0.01) ~ 0.202. Need ~5 to hit 1.0, but excluding self
        // leaves 4 units; accumulated weight < 1.0 probably.
        let weights: Vec<f64> = vec![0.2; 5];
        let w_slice: &[f64] = &weights;
        let mut s = WeightedSearcher::from_unit(tree.data(), 2, 0.01).unwrap();
        s.search(&tree, &w_slice).unwrap();

        // 4 other units each contributing ~0.202, sum ~= 0.808 < 1.0
        assert!(s.total_weight() < 1.0);
        // All 4 non-self neighbours should be included
        assert_eq!(s.neighbours().len(), 4);
    }
}
