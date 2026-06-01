// Copyright (C) 2026 Wilmer Prentius.

// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! kd-tree searchers
//!
//! Searchers are used to traverse a kd-tree, looking for the node of a search unit or point.

use std::num::NonZeroUsize;

pub use neighbour::NeighbourSlice;
use neighbour::{
    Neighbour,
    WeightedNeighbour,
};
use num_traits::ConstZero;

use super::Tree;
use crate::number_traits::Number;
use crate::spatial::PointSet;

pub mod neighbour {
    use std::cmp::Ordering;

    use crate::number_traits::Number;

    /// A neighbouring unit, some squared euclidean distance away
    ///
    /// A neighbour is equal to another if their distances are the same.
    #[derive(Copy, Clone, Debug)]
    pub struct Neighbour<Id, Value> {
        /// Id of unit
        id: Id,
        /// Squared euclidean distance
        distance: Value,
    }
    impl<Id, Value> Neighbour<Id, Value> {
        /// Returns the neighbour id.
        #[inline]
        pub fn id(&self) -> Id
        where
            Id: Copy,
        {
            self.id
        }
        /// Returns the squared euclidean distance to the neighbour
        #[inline]
        pub fn distance(&self) -> Value
        where
            Value: Copy,
        {
            self.distance
        }
        /// Constructs a new neighbour
        #[inline]
        pub fn new(id: Id, distance: Value) -> Self { Self { id, distance } }
    }

    impl<Id, Value> Ord for Neighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering { Value::compare(&self.distance, &other.distance) }
    }
    impl<Id, Value> PartialOrd for Neighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }
    impl<Id, Value> PartialEq for Neighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn eq(&self, other: &Self) -> bool { self.cmp(other).is_eq() }
    }
    impl<Id, Value> Eq for Neighbour<Id, Value> where Value: Number {}

    /// A weighted neighbouring unit, some squared euclidean distance away
    ///
    /// A weighted neighbour is equal to another if their distances and weights are the same.
    /// A weighted neighbour is sorted before another if its distance is smaller, or its distance is
    /// equal but its weight is smaller.
    #[derive(Copy, Clone, Debug)]
    pub struct WeightedNeighbour<Id, Value> {
        /// The neighbour
        neighbour: Neighbour<Id, Value>,
        /// The weight
        weight: f64,
    }

    impl<Id, Value> WeightedNeighbour<Id, Value> {
        /// Returns the id of the neighbour.
        #[inline]
        pub fn id(&self) -> Id
        where
            Id: Copy,
        {
            self.neighbour.id
        }
        /// Returns the squared euclidean distance to the neighbour
        #[inline]
        pub fn distance(&self) -> Value
        where
            Value: Copy,
        {
            self.neighbour.distance
        }
        /// Returns the weight of the neighbour.
        #[inline]
        pub fn weight(&self) -> f64 { self.weight }
        /// Constructs a new weighted neighbour
        #[inline]
        pub fn new(id: Id, distance: Value, weight: f64) -> Self {
            let neighbour = Neighbour::new(id, distance);
            Self { neighbour, weight }
        }
    }
    impl<Id, Value> Ord for WeightedNeighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.neighbour
                .cmp(&other.neighbour)
                .then(self.weight().total_cmp(&other.weight()))
        }
    }
    impl<Id, Value> PartialOrd for WeightedNeighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
    }
    impl<Id, Value> PartialEq for WeightedNeighbour<Id, Value>
    where
        Value: Number,
    {
        #[inline]
        fn eq(&self, other: &Self) -> bool { self.cmp(other).is_eq() }
    }
    impl<Id, Value> Eq for WeightedNeighbour<Id, Value> where Value: Number {}

    pub trait NeighbourSlice<Id> {
        fn to_neighbour_id_iter(&self) -> impl Iterator<Item = Id>;
        #[inline]
        fn to_neighbour_ids(&self) -> Box<[Id]> {
            self.to_neighbour_id_iter()
                .collect::<Vec<_>>()
                .into_boxed_slice()
        }
        #[inline]
        fn contains_id(&self, id: Id) -> bool
        where
            Id: Eq,
        {
            self.to_neighbour_id_iter().any(|nid| nid == id)
        }
    }
    impl<Id, Value> NeighbourSlice<Id> for [Neighbour<Id, Value>]
    where
        Id: Copy,
        Value: Copy,
    {
        #[inline]
        fn to_neighbour_id_iter(&self) -> impl Iterator<Item = Id> {
            self.iter().map(Neighbour::id)
        }
    }
    impl<Id, Value> NeighbourSlice<Id> for [WeightedNeighbour<Id, Value>]
    where
        Id: Copy,
        Value: Copy,
    {
        #[inline]
        fn to_neighbour_id_iter(&self) -> impl Iterator<Item = Id> {
            self.iter().map(WeightedNeighbour::id)
        }
    }
}

/// A trait for searching in a kd-[`Tree`]
pub trait TreeSearcher<P>
where
    P: PointSet,
{
    /// Returns a reference to the searching point.
    fn point(&self) -> &[P::Value];
    /// Returns `true` if the search does not need to visit a node.
    fn is_satisfied(&self, distance: P::Value) -> bool;
    /// A function called on the leafs of each visited node.
    fn visit_leaf(&mut self, data: &P, leaf_units: &[P::Id]);
}

/// The search point of the [`TreeSearcher`]
#[must_use]
#[derive(Clone, Debug)]
pub struct SearchPoint<P>
where
    P: PointSet,
{
    /// Search point
    point: Box<[P::Value]>,
    /// Potential id of the point
    unit: Option<P::Id>,
}
impl<P> SearchPoint<P>
where
    P: PointSet,
{
    /// Returns the search point
    #[must_use]
    #[inline]
    pub fn point(&self) -> &[P::Value] { &self.point }
    /// Returns the potential id of the search point
    #[must_use]
    #[inline]
    pub fn unit(&self) -> Option<P::Id> { self.unit }
    /// Returns `true` if the id of `other` matches the potential id of the search point.
    #[must_use]
    #[inline]
    pub fn is_unit(&self, other: P::Id) -> bool { self.unit.is_some_and(|u| u == other) }
    /// Constructs a new, uninitialized search point
    #[inline]
    pub fn new(dim: NonZeroUsize) -> Self {
        Self {
            point: vec![P::Value::ZERO; dim.get()].into_boxed_slice(),
            unit: None,
        }
    }
    /// Constructs a new search point from a unit id
    /// Returns `None` if the unit does not exist in the `data`.
    #[inline]
    pub fn from_unit(data: &P, unit: P::Id) -> Option<Self> {
        Some(Self {
            point: data.to_boxed_slice(unit)?,
            unit: Some(unit),
        })
    }
    /// Constructs a new search point from a point slice
    /// Returns `None` if the point is empty.
    #[inline]
    pub fn from_slice(point: &[P::Value]) -> Option<Self> {
        (!point.is_empty()).then(|| Self {
            point: point.into(),
            unit: None,
        })
    }
    /// Sets the search point according to the `unit` from `data`.
    /// Returns `None` if the unit does not exist in the `data`.
    #[must_use]
    #[inline]
    pub fn set_from_unit(&mut self, data: &P, unit: P::Id) -> Option<()> {
        self.point = data.to_boxed_slice(unit)?;
        self.unit = Some(unit);
        Some(())
    }
    /// Sets the search point from a point slice.
    /// Returns `None` if the point dimensionaliy does not match the corrent search point dim.
    #[must_use]
    #[inline]
    pub fn set_from_slice(&mut self, point: &[P::Value]) -> Option<()> {
        (self.point.len() == point.len()).then(|| {
            self.point = point.into();
            self.unit = None;
        })
    }
}

/// Searching the nearest neighbour of a unit.
#[must_use]
#[derive(Clone, Debug)]
pub struct NearestNeighbourSearcher<P>
where
    P: PointSet,
{
    /// Search point
    point: SearchPoint<P>,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`]
    neighbours: Vec<Neighbour<P::Id, P::Value>>,
}
impl<P> NearestNeighbourSearcher<P>
where
    P: PointSet,
{
    /// Constructs a new, uninitialized, 1NN-searcher.
    #[inline]
    pub fn new(data: &P) -> Self {
        Self {
            point: SearchPoint::new(data.dim()),
            neighbours: Vec::with_capacity(6),
        }
    }
    /// Constructs a new 1NN-searcher from a `unit` according to `data`.
    /// Returns `None` if the unit does not exist in `data`.
    #[inline]
    pub fn from_unit(data: &P, unit: P::Id) -> Option<Self> {
        Some(Self {
            point: SearchPoint::from_unit(data, unit)?,
            neighbours: Vec::with_capacity(6),
        })
    }
    /// Constructs a new 1NN-searcher from a point.
    /// Returns `None` if the point is empty.
    #[inline]
    pub fn from_slice(point: &[P::Value]) -> Option<Self> {
        Some(Self {
            point: SearchPoint::from_slice(point)?,
            neighbours: Vec::with_capacity(6),
        })
    }
    /// Sets the search point according to the `unit` from `data`.
    /// Returns `None` if the unit does not exist in the `data`.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_unit(&mut self, data: &P, unit: P::Id) -> Option<&mut Self> {
        self.point.set_from_unit(data, unit)?;
        self.neighbours.clear();
        Some(self)
    }
    /// Sets the search point from a point slice.
    /// Returns `None` if the point dimensionaliy does not match the corrent search point dim.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[P::Value]) -> Option<&mut Self> {
        self.point.set_from_slice(point)?;
        self.neighbours.clear();
        Some(self)
    }
    /// Finds the 1NN of the search point.
    /// In cases of ties, all nearest neighbours are added.
    #[must_use]
    #[inline]
    pub fn search(&mut self, tree: &Tree<P>) -> Option<()> {
        self.neighbours.clear();
        tree.iterate_leafs_by(self)
    }
    /// Returns the neighbours of the latest search.
    #[must_use]
    #[inline]
    pub fn neighbours(&self) -> &[Neighbour<P::Id, P::Value>] { &self.neighbours }
    /// Returns the maximum squared euclidean distance of the neighbours in the latest search.
    /// Returns `None` if no neighbours were found.
    #[must_use]
    #[inline]
    pub fn max_distance(&self) -> Option<P::Value> {
        self.neighbours.last().map(Neighbour::distance)
    }
}
impl<P> TreeSearcher<P> for NearestNeighbourSearcher<P>
where
    P: PointSet,
{
    #[must_use]
    #[inline]
    fn point(&self) -> &[P::Value] { self.point.point() }
    #[must_use]
    #[inline]
    fn is_satisfied(&self, distance: P::Value) -> bool {
        // Satisfied only if enough units AND a potential unit is not further away
        // !self.neighbours.is_empty() && self.max_distance() < distance.powi(2)
        self.max_distance()
            .is_some_and(|md| md < distance * distance)
    }
    #[inline]
    fn visit_leaf(&mut self, data: &P, leaf_units: &[P::Id]) {
        let mut current_max = self
            .max_distance()
            .unwrap_or(<P::Value as Number>::max_value());
        for &id in leaf_units {
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
/// If `k == 1`, use [`NearestNeighbourSearcher`] instead.
#[must_use]
#[derive(Clone, Debug)]
pub struct KNearestNeighbourSearcher<P>
where
    P: PointSet,
{
    /// Search point
    point: SearchPoint<P>,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`]
    neighbours: Vec<Neighbour<P::Id, P::Value>>,
    /// The `k` number of neighbours to search for.
    nominal_size: NonZeroUsize,
}
impl<P> KNearestNeighbourSearcher<P>
where
    P: PointSet,
{
    /// Constructs a new, uninitialized, kNN-searcher.
    #[inline]
    pub fn new(k: NonZeroUsize, data: &P) -> Self {
        Self {
            point: SearchPoint::new(data.dim()),
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        }
    }
    /// Constructs a new kNN-searcher from a `unit` according to `data`.
    /// Returns `None` if the unit does not exist in `data`.
    #[inline]
    pub fn from_unit(k: NonZeroUsize, data: &P, unit: P::Id) -> Option<Self> {
        Some(Self {
            point: SearchPoint::from_unit(data, unit)?,
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        })
    }
    /// Constructs a new kNN-searcher from a point.
    /// Returns `None` if the point is empty.
    #[inline]
    pub fn from_slice(k: NonZeroUsize, point: &[P::Value]) -> Option<Self> {
        Some(Self {
            point: SearchPoint::from_slice(point)?,
            neighbours: Vec::with_capacity(k.get() + 6),
            nominal_size: k,
        })
    }
    /// Sets the `k`, the number of nearest neighbours to search for.
    #[inline]
    pub fn set_nominal_size(&mut self, k: NonZeroUsize) -> &mut Self {
        self.nominal_size = k;
        self
    }
    /// Sets the search point according to the `unit` from `data`.
    /// Returns `None` if the unit does not exist in the `data`.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_unit(&mut self, data: &P, unit: P::Id) -> Option<&mut Self> {
        self.point.set_from_unit(data, unit)?;
        self.neighbours.clear();
        Some(self)
    }
    /// Sets the search point from a point slice.
    /// Returns `None` if the point dimensionaliy does not match the corrent search point dim.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[P::Value]) -> Option<&mut Self> {
        self.point.set_from_slice(point)?;
        self.neighbours.clear();
        Some(self)
    }
    /// Finds the kNN of the search point.
    /// In cases of ties, all nearest neighbours are added.
    #[must_use]
    #[inline]
    pub fn search(&mut self, tree: &Tree<P>) -> Option<()> {
        self.neighbours.clear();
        tree.iterate_leafs_by(self)
    }
    /// Returns the neighbours of the latest search.
    #[must_use]
    #[inline]
    pub fn neighbours(&self) -> &[Neighbour<P::Id, P::Value>] { &self.neighbours }
    /// Returns the maximum squared euclidean distance of the neighbours in the latest search.
    /// Returns `None` if no neighbours were found.
    #[must_use]
    #[inline]
    pub fn max_distance(&self) -> Option<P::Value> {
        self.neighbours.last().map(Neighbour::distance)
    }
}
impl<P> TreeSearcher<P> for KNearestNeighbourSearcher<P>
where
    P: PointSet,
{
    #[must_use]
    #[inline]
    fn point(&self) -> &[P::Value] { self.point.point() }
    #[must_use]
    #[inline]
    fn is_satisfied(&self, distance: P::Value) -> bool {
        // Satisfied only if enough units AND a potential unit is not further away
        // self.neighbours.len() >= self.nominal_size.get() && self.max_distance() <
        // distance.powi(2)
        self.neighbours.len() >= self.nominal_size.get()
            && self
                .max_distance()
                .is_some_and(|md| md < distance * distance)
    }
    #[inline]
    fn visit_leaf(&mut self, data: &P, leaf_units: &[P::Id]) {
        let original_len = self.neighbours.len();

        // Default to 0.0 b/c trick below.
        // self.neighbours is assumed to be sorted by distance.
        let mut current_max = self
            .max_distance()
            .unwrap_or(<P::Value as Number>::max_value());
        for &id in leaf_units {
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

/// Searching the nearest neighbour of a unit, until the total weight of the neighbours sum to 1.0.
#[must_use]
#[derive(Clone, Debug)]
pub struct WeightedSearcher<P>
where
    P: PointSet,
{
    /// Search point
    point: SearchPoint<P>,
    /// The weight of the search point
    point_weight: f64,
    /// The neighbours, sorted ascending by distance to the [`SearchPoint`], where lower weights are
    /// sorted before higher weights in case of ties.
    neighbours: Vec<WeightedNeighbour<P::Id, P::Value>>,
    /// The total weight of the neighbours
    total_weight: f64,
}
impl<P> WeightedSearcher<P>
where
    P: PointSet,
{
    /// Constructs a new, uninitialized, wNN-searcher.
    #[inline]
    pub fn new(data: &P) -> Self {
        Self {
            point: SearchPoint::new(data.dim()),
            point_weight: 0.5,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        }
    }
    /// Constructs a new wNN-searcher from a `unit` according to `data`.
    /// Returns `None` if the unit does not exist in `data`.
    #[inline]
    pub fn from_unit(data: &P, unit: P::Id, weight: f64) -> Option<Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        Some(Self {
            point: SearchPoint::from_unit(data, unit)?,
            point_weight: weight,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        })
    }
    /// Constructs a new wNN-searcher from a point.
    /// Returns `None` if the point is empty.
    #[inline]
    pub fn from_slice(point: &[P::Value], weight: f64) -> Option<Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        Some(Self {
            point: SearchPoint::from_slice(point)?,
            point_weight: weight,
            neighbours: Vec::with_capacity(6),
            total_weight: 0.0,
        })
    }
    /// Sets the search point according to the `unit` from `data`.
    /// Returns `None` if the unit does not exist in the `data`.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_unit(&mut self, data: &P, unit: P::Id, weight: f64) -> Option<&mut Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        self.point.set_from_unit(data, unit)?;
        self.point_weight = weight;
        self.neighbours.clear();
        self.total_weight = 0.0;
        Some(self)
    }
    /// Sets the search point from a point slice.
    /// Returns `None` if the point dimensionaliy does not match the corrent search point dim.
    ///
    /// Clears the previous search.
    #[inline]
    pub fn reset_from_slice(&mut self, point: &[P::Value], weight: f64) -> Option<&mut Self> {
        if !(0.0 < weight && weight < 1.0) {
            return None;
        }
        self.point.set_from_slice(point)?;
        self.point_weight = weight;
        self.neighbours.clear();
        self.total_weight = 0.0;
        Some(self)
    }
    /// Finds the wNN of the search point.
    /// In cases of ties, all nearest neighbours are added.
    #[must_use]
    #[inline]
    pub fn search<W>(&mut self, tree: &Tree<P>, weights: &W) -> Option<()>
    where
        W: WeightCollection<P::Id>,
    {
        if !(0.0 < self.point_weight && self.point_weight < 1.0) {
            return None;
        }
        self.neighbours.clear();
        self.total_weight = 0.0;
        let mut searcher = WeightedSearcherWrapper::new(self, weights);
        tree.iterate_leafs_by(&mut searcher)
    }
    /// Returns the neighbours of the latest search.
    #[must_use]
    #[inline]
    pub fn neighbours(&self) -> &[WeightedNeighbour<P::Id, P::Value>] { &self.neighbours }
    /// Returns the sum of the weight of the neighbours
    #[must_use]
    #[inline]
    pub fn total_weight(&self) -> f64 { self.total_weight }
    /// Returns the weight of the search point
    #[must_use]
    #[inline]
    pub fn point_weight(&self) -> f64 { self.point_weight }
    /// Returns the maximum squared euclidean distance of the neighbours in the latest search.
    /// Returns `None` if no neighbours were found.
    #[must_use]
    #[inline]
    pub fn max_distance(&self) -> Option<P::Value> {
        self.neighbours.last().map(WeightedNeighbour::distance)
    }
    /// Helper for calculating the potential weight, where other is assumed to be a probability.
    #[must_use]
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

pub trait WeightCollection<Id> {
    #[must_use]
    fn try_get_weight(&self, id: Id) -> Option<f64>;
    #[must_use]
    #[inline]
    fn get_weight(&self, id: Id) -> f64 { self.try_get_weight(id).expect("id to exist") }
}
impl WeightCollection<usize> for &[f64] {
    #[inline]
    fn try_get_weight(&self, id: usize) -> Option<f64> { self.get(id).copied() }
    #[inline]
    fn get_weight(&self, id: usize) -> f64 { self[id] }
}

/// Wrapper for the [`WeightedSearcher`]
/// Needed as weights cannot be borrowed in [`WeightedSearcher`], as they will be mutated in between
/// searches.
#[must_use]
#[derive(Debug)]
struct WeightedSearcherWrapper<'borrow, P, W>
where
    P: PointSet,
{
    /// The (public) searcher
    searcher: &'borrow mut WeightedSearcher<P>,
    /// The weights used
    weights: &'borrow W,
}
impl<'borrow, P, W> WeightedSearcherWrapper<'borrow, P, W>
where
    P: PointSet,
{
    /// Constructs a new wrapper around weigthed searcher
    fn new(searcher: &'borrow mut WeightedSearcher<P>, weights: &'borrow W) -> Self {
        Self { searcher, weights }
    }
}
impl<P, W> TreeSearcher<P> for WeightedSearcherWrapper<'_, P, W>
where
    P: PointSet,
    W: WeightCollection<P::Id>,
{
    #[must_use]
    #[inline]
    fn point(&self) -> &[P::Value] { self.searcher.point.point() }
    #[must_use]
    #[inline]
    fn is_satisfied(&self, distance: P::Value) -> bool {
        self.searcher.total_weight >= 1.0
            && self
                .searcher
                .max_distance()
                .is_some_and(|md| md < distance * distance)
    }
    fn visit_leaf(&mut self, data: &P, leaf_units: &[P::Id]) {
        let original_len = self.searcher.neighbours.len();

        // Default to 0.0 b/c trick below
        let mut current_max = self
            .searcher
            .max_distance()
            .unwrap_or(<P::Value as ConstZero>::ZERO);
        for &id in leaf_units {
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

    use super::*;
    use crate::matrix::Matrix;
    use crate::test_utils::*;

    // Helper to create a 2D Matrix PointSet
    fn setup_matrix() -> Matrix<f64> {
        // Points: (0,0), (1,0), (0,1), (1,1)
        let data = vec![
            0.0, 1.0, 0.0, 1.0, // Dim 0
            0.0, 0.0, 1.0, 1.0, // Dim 1
        ];
        Matrix::new(data, nz(4)).unwrap()
    }

    // --- NearestNeighbourSearcher Tests ---

    #[test]
    fn test_nn_search_basic() {
        let mat = setup_matrix();
        let mut searcher = NearestNeighbourSearcher::from_slice(&[0.1, 0.1]).unwrap();

        // Mock visit to all units
        searcher.visit_leaf(&mat, &[0, 1, 2, 3]);

        let neighbours = searcher.neighbours();
        assert_eq!(neighbours.len(), 1);
        assert_eq!(neighbours[0].id(), 0); // (0,0) is closest to (0.1, 0.1)
    }

    #[test]
    fn test_nn_tie_handling() {
        let mat = setup_matrix();
        // Search from center (0.5, 0.5)
        let mut searcher = NearestNeighbourSearcher::from_slice(&[0.5, 0.5]).unwrap();

        searcher.visit_leaf(&mat, &[0, 1, 2, 3]);

        // All 4 points are equidistant (dist_sq = 0.5)
        let neighbours = searcher.neighbours();
        assert_eq!(neighbours.len(), 4);
    }

    // --- KNearestNeighbourSearcher Tests ---

    #[test]
    fn test_knn_basic_limit() {
        let mat = setup_matrix();
        let k = nz(2);
        let mut searcher = KNearestNeighbourSearcher::from_slice(k, &[0.4, 0.0]).unwrap();

        searcher.visit_leaf(&mat, &[0, 1, 2, 3]);

        let neighbours = searcher.neighbours();
        // Should find exactly 2 neighbours
        assert_eq!(neighbours.len(), 2);
        assert_eq!(neighbours[0].id(), 0); // dist 0.4
        assert_eq!(neighbours[1].id(), 1); // dist 0.6
    }

    #[test]
    fn test_knn_tie_at_boundary() {
        let mat = setup_matrix();
        let k = nz(1); // Ask for 1
        // Search from (0.5, 0.0). Units (0,0) and (1,0) are both dist_sq 0.25
        let mut searcher = KNearestNeighbourSearcher::from_slice(k, &[0.5, 0.0]).unwrap();

        searcher.visit_leaf(&mat, &[0, 1, 2, 3]);

        let neighbours = searcher.neighbours();
        // Should keep both because they tie for the k-th position
        assert_eq!(neighbours.len(), 2);
    }

    // --- WeightedSearcher Tests ---

    #[test]
    fn test_weighted_search_accumulation() {
        let mat = setup_matrix();
        // Weights for units 0, 1, 2, 3
        let weights: &[f64] = &[0.2, 0.2, 0.2, 0.2];

        // Search point weight = 0.5
        // calculate_weight for 0.2: 0.2 / (1.0 - 0.5) = 0.4
        let mut searcher = WeightedSearcher::from_slice(&[0.0, 0.0], 0.5).unwrap();
        let mut wrapper = WeightedSearcherWrapper::new(&mut searcher, &weights);

        wrapper.visit_leaf(&mat, &[0, 1, 2, 3]);

        // Total weight needed is 1.0. Each unit provides 0.4.
        // Needs 3 units (0.4 * 3 = 1.2) to satisfy >= 1.0
        assert!(searcher.total_weight() >= 1.0);
        assert_eq!(searcher.neighbours().len(), 3);
    }

    #[test]
    fn test_weighted_search_tie_behavior() {
        let mat = setup_matrix();
        let weights: &[f64] = &[0.1, 0.1, 0.1, 0.1];

        // Search from (0.5, 0.5) where all points tie for distance
        let mut searcher = WeightedSearcher::from_slice(&[0.5, 0.5], 0.8).unwrap();
        let mut wrapper = WeightedSearcherWrapper::new(&mut searcher, &weights);

        wrapper.visit_leaf(&mat, &[0, 1, 2, 3]);

        // Because they all have the same distance, even if the weight
        // threshold is met early, all equidistant points must be kept
        assert_eq!(searcher.neighbours().len(), 4);
    }
}
