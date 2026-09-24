// Copyright (C) 2026 Wilmer Prentius.

// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.

// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Defines structures for storing neighbours in searchers.

use std::cmp::Ordering;

use crate::utils::Number;

/// Provides a view into the views and distances to a neighbour.
pub trait NeighbourView {
    /// The type of the identifier
    type ID;
    /// The type of the distance value
    type DIST;
    /// Returns a reference to the id of the neighbour
    #[must_use]
    fn id(&self) -> &Self::ID;
    /// Returns a reference to the squared euclidean distance to the neighbour
    #[must_use]
    fn distance(&self) -> &Self::DIST;
}

/// A neighbouring unit, some squared euclidean distance away.
///
/// A neighbour is equal to another if their distances are the same.
#[must_use]
#[derive(Copy, Clone, Debug)]
pub struct Neighbour<ID, DIST> {
    /// Id of unit
    id: ID,
    /// Squared euclidean distance
    distance: DIST,
}
impl<ID, DIST> Neighbour<ID, DIST> {
    /// Constructs a new neighbour.
    #[inline]
    pub fn new(id: ID, distance: DIST) -> Self { Self { id, distance } }
}
impl<ID, DIST> NeighbourView for Neighbour<ID, DIST> {
    type ID = ID;
    type DIST = DIST;
    #[inline]
    fn id(&self) -> &ID { &self.id }
    #[inline]
    fn distance(&self) -> &DIST { &self.distance }
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

/// A weighted neighbouring unit, some squared euclidean distance away.
///
/// A weighted neighbour is equal to another if their distances and weights are the same.
/// A weighted neighbour is sorted before another if its distance is smaller, or its distance is
/// equal but its weight is smaller.
#[derive(Copy, Clone, Debug)]
pub struct WeightedNeighbour<ID, DIST> {
    /// The neighbour
    neighbour: Neighbour<ID, DIST>,
    /// The weight
    weight: f64,
}

impl<ID, DIST> WeightedNeighbour<ID, DIST> {
    /// Returns the weight of the neighbour.
    #[inline]
    pub fn weight(&self) -> f64 { self.weight }
    /// Constructs a new weighted neighbour.
    #[inline]
    pub fn new(id: ID, distance: DIST, weight: f64) -> Self {
        let neighbour = Neighbour::new(id, distance);
        Self { neighbour, weight }
    }
}
impl<ID, DIST> NeighbourView for WeightedNeighbour<ID, DIST> {
    type ID = ID;
    type DIST = DIST;
    #[inline]
    fn id(&self) -> &ID { &self.neighbour.id }
    #[inline]
    fn distance(&self) -> &DIST { &self.neighbour.distance }
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
