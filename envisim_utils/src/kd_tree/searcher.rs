// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

use super::Node;
use crate::{InputError, Matrix, Probabilities};
use std::cmp::Ordering;
use std::num::NonZeroUsize;

pub(super) trait TreeSearcher {
    fn unit(&self) -> &[f64];
    fn max_distance(&self) -> Option<f64>;
    fn is_satisfied(&self) -> bool;
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &Matrix);
}

/// A struct used for searching the nearest neighbour of a unit.
pub struct Searcher {
    unit_id: usize,
    unit: Vec<f64>,
    neighbours: Vec<usize>,
    distances: Vec<f64>,
    n_neighbours: NonZeroUsize,
}

impl TreeSearcher for Searcher {
    #[inline]
    fn unit(&self) -> &[f64] {
        &self.unit
    }
    #[inline]
    fn max_distance(&self) -> Option<f64> {
        self.neighbours.last().map(|&id| self.distances[id])
    }
    #[inline]
    fn is_satisfied(&self) -> bool {
        self.neighbours.len() >= self.n_neighbours.get()
    }
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &Matrix) {
        if ids.is_empty() {
            return;
        }

        if self.n_neighbours.get() == 1 {
            self.assess_units_1(ids, data);
            return;
        }

        let original_length = self.neighbours.len();
        self.assess_units(ids, data);

        // If store size hasn't changed, we haven't added any new units
        if self.neighbours.len() == original_length {
            return;
        }

        self.sort_neighbours();
        self.truncate_neighbours();
    }
}

impl Searcher {
    /// Constructs a new k-d tree searcher, for finding `n_neighbours` number of neighbours
    #[inline]
    pub fn new(node: &Node, n_neighbours: NonZeroUsize) -> Searcher {
        let data = node.data();

        Self {
            unit_id: usize::MAX,
            unit: vec![0.0f64; data.ncol()],
            neighbours: Vec::<usize>::with_capacity(data.nrow()),
            distances: vec![0.0f64; data.nrow()],
            n_neighbours,
        }
    }
    /// Constructs a new k-d tree searcher, for finding closest neighbour
    #[inline]
    pub fn new_1(node: &Node) -> Searcher {
        Self::new(node, unsafe { NonZeroUsize::new_unchecked(1) })
    }
    /// Finds the neighbours of a unit positioned at `unit`
    #[inline]
    pub fn find_neighbours(&mut self, node: &Node, unit: &[f64]) -> Result<(), InputError> {
        self.set_unit_from_iter(unit.iter(), usize::MAX)?;
        node.find_neighbours(self);
        Ok(())
    }
    /// Finds the neighbours of a unit in the data matrix
    #[inline]
    pub fn find_neighbours_of_id(&mut self, node: &Node, idx: usize) -> Result<(), InputError> {
        self.set_unit_from_iter(node.data().row_iter(idx), idx)?;
        node.find_neighbours(self);
        Ok(())
    }
    /// Finds the neighbours of a unit positioned at the vector constructed by the iterator
    #[inline]
    pub fn find_neighbours_of_iter<'a, I>(&mut self, node: &Node, iter: I) -> Result<(), InputError>
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        self.set_unit_from_iter(iter, usize::MAX)?;
        node.find_neighbours(self);
        Ok(())
    }
    #[inline]
    fn set_unit_from_iter<'a, I>(&mut self, iter: I, idx: usize) -> Result<(), InputError>
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        InputError::check_sizes(iter.len(), self.unit.len())?;

        self.unit_id = idx;
        self.unit.iter_mut().zip(iter).for_each(|(a, b)| *a = *b);
        self.reset();
        Ok(())
    }
    /// Set the number of neighbours to search for.
    #[inline]
    pub fn set_n_neighbours(&mut self, n: NonZeroUsize) {
        self.n_neighbours = n;
    }

    /// Get the list of found neighbours
    #[inline]
    pub fn neighbours(&self) -> &[usize] {
        &self.neighbours
    }

    /// Reset the list of found neighbours
    #[inline]
    pub fn reset(&mut self) {
        self.neighbours.clear();
    }
    #[inline]
    fn add(&mut self, idx: usize, distance: f64) {
        self.neighbours.push(idx);
        self.distances[idx] = distance;
    }
    #[inline]
    fn assess_units_1(&mut self, ids: &[usize], data: &Matrix) {
        let mut current_max = self.max_distance().unwrap_or(f64::INFINITY);

        ids.iter().for_each(|&id| {
            if self.unit_id == id {
                return;
            }

            let distance = data.distance_to_row(id, &self.unit);

            if distance < current_max {
                self.reset();
                self.add(id, distance);
                current_max = distance;
            } else if distance == current_max {
                self.add(id, distance);
            }
        });
    }
    #[inline]
    fn assess_units(&mut self, ids: &[usize], data: &Matrix) {
        // The case of when the store isn't filled yet
        // node_max will store the max _added_ distance from the node
        let mut node_max: f64 = self.max_distance().unwrap_or(0.0);

        ids.iter().for_each(|&id| {
            if self.unit_id == id {
                return;
            }

            let distance = data.distance_to_row(id, &self.unit);

            // We should add a unit only in two circumstances:
            // - if the unit is closer than the current largest dist
            // - if the store still has room

            if distance <= node_max {
                self.add(id, distance);
            } else if self.neighbours.len() < self.n_neighbours.get() {
                self.add(id, distance);
                node_max = distance;
            }
        });
    }
    /// Get the squared euclidean distance to the `k`th neighbour
    #[inline]
    pub fn distance_k(&self, k: usize) -> f64 {
        self.distances[self.neighbours[k]]
    }
    #[inline]
    fn sort_neighbours(&mut self) {
        self.neighbours
            .sort_unstable_by(|a, b| self.distances[*a].partial_cmp(&self.distances[*b]).unwrap());
    }
    #[inline]
    fn truncate_neighbours(&mut self) {
        let mut i: usize = 1;
        let len: usize = self.neighbours.len();

        while i < len {
            if i >= self.n_neighbours.get() && self.distance_k(i - 1) < self.distance_k(i) {
                break;
            }

            i += 1;
        }

        self.neighbours.truncate(i);
    }
}

pub struct SearcherWeighted {
    searcher: Searcher,
    weights: Vec<f64>,
}

impl SearcherWeighted {
    /// Constructs a new k-d tree searcher, for finding a (probability) weighted number of
    /// neighbours.
    #[inline]
    pub fn new(node: &Node) -> Self {
        Self {
            searcher: Searcher::new(node, unsafe { NonZeroUsize::new_unchecked(1) }),
            weights: vec![0.0f64; node.data().nrow()],
        }
    }
    /// Finds the neighbours of a unit positioned at `unit`, with a specified probability
    #[inline]
    pub fn find_neighbours(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        unit: &[f64],
        prob: f64,
    ) -> Result<(), InputError> {
        self.searcher.set_unit_from_iter(unit.iter(), usize::MAX)?;
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities,
            total_weight: 0.0,
            unit_prob: prob,
        };
        node.find_neighbours(&mut tree_searcher);
        Ok(())
    }
    /// Finds the neighbours of a unit in the data matrix
    #[inline]
    pub fn find_neighbours_of_id(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        idx: usize,
    ) -> Result<(), InputError> {
        self.searcher
            .set_unit_from_iter(node.data().row_iter(idx), idx)?;
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities,
            total_weight: 0.0,
            unit_prob: probabilities[idx],
        };
        node.find_neighbours(&mut tree_searcher);
        Ok(())
    }
    /// Finds the neighbours of a unit positioned at the vector constructed by the iterator
    #[inline]
    pub fn find_neighbours_of_iter<'a, I>(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        iter: I,
        prob: f64,
    ) -> Result<(), InputError>
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        self.searcher.set_unit_from_iter(iter, usize::MAX)?;
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities,
            total_weight: 0.0,
            unit_prob: prob,
        };
        node.find_neighbours(&mut tree_searcher);
        Ok(())
    }

    /// Get the list of found neighbours
    #[inline]
    pub fn neighbours(&self) -> &[usize] {
        &self.searcher.neighbours
    }

    /// Reset the list of found neighbours
    #[inline]
    pub fn reset(&mut self) {
        self.searcher.neighbours.clear();
    }

    /// Get the squared euclidean distance to the `k`th neighbour
    #[inline]
    pub fn distance_k(&self, k: usize) -> f64 {
        self.searcher.distance_k(k)
    }
    /// Get the assigned weight of the `k`th neighbour
    #[inline]
    pub fn weight_k(&self, k: usize) -> f64 {
        self.weights[self.searcher.neighbours[k]]
    }
    /// Sort a section of the neighbours by distance and weight
    pub fn sort_by_weight(&mut self, from: usize, to: usize) {
        self.searcher.neighbours[from..to].sort_unstable_by(|&a, &b| {
            if self.searcher.distances[a] < self.searcher.distances[b] {
                return Ordering::Less;
            } else if self.searcher.distances[a] > self.searcher.distances[b] {
                return Ordering::Greater;
            }

            if self.weights[a] < self.weights[b] {
                return Ordering::Less;
            } else if self.weights[a] > self.weights[b] {
                return Ordering::Greater;
            }

            Ordering::Equal
        });
    }
}

pub(super) struct TreeSearcherWeighted<'a> {
    searcher: &'a mut SearcherWeighted,
    probabilities: &'a Probabilities,
    total_weight: f64,
    unit_prob: f64,
}

impl<'a> TreeSearcherWeighted<'a> {
    #[inline]
    fn base(&self) -> &Searcher {
        &self.searcher.searcher
    }
    #[inline]
    fn base_mut(&mut self) -> &mut Searcher {
        &mut self.searcher.searcher
    }

    #[inline]
    fn add(&mut self, idx: usize, distance: f64) -> f64 {
        self.base_mut().add(idx, distance);
        let weight = self.probabilities.weight_to(self.unit_prob, idx);
        self.searcher.weights[idx] = weight;
        weight
    }

    fn assess_units(&mut self, ids: &[usize], data: &Matrix) {
        // The case of when the store isn't filled yet
        // node_max will store the max _added_ distance from the node
        let mut node_max: f64 = self.max_distance().unwrap_or(0.0);

        ids.iter().for_each(|&id| {
            if self.base().unit_id == id {
                return;
            }

            let distance = data.distance_to_row(id, &self.base().unit);

            // We should add a unit only in two circumstances:
            // - if the unit is closer than the current largest dist
            // - if the store still has room

            if distance <= node_max {
                self.total_weight += self.add(id, distance);
            } else if self.total_weight < 1.0 {
                self.total_weight += self.add(id, distance);
                node_max = distance;
            }
        });
    }
    fn truncate_neighbours(&mut self) {
        let len: usize = self.base().neighbours.len();

        if len == 0 {
            self.total_weight = 0.0;
            return;
        } else {
            self.total_weight = self.searcher.weight_k(0);
        }

        let mut i: usize = 1;

        while i < len {
            if self.total_weight >= 1.0 && self.base().distance_k(i - 1) < self.base().distance_k(i)
            {
                break;
            }

            self.total_weight += self.searcher.weight_k(i);
            i += 1;
        }

        self.base_mut().neighbours.truncate(i);
    }
}

impl<'a> TreeSearcher for TreeSearcherWeighted<'a> {
    fn unit(&self) -> &[f64] {
        self.base().unit()
    }
    fn max_distance(&self) -> Option<f64> {
        self.base().max_distance()
    }
    fn is_satisfied(&self) -> bool {
        self.total_weight >= 1.0
    }
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &Matrix) {
        if ids.is_empty() {
            return;
        }

        let original_length = self.base().neighbours.len();
        self.assess_units(ids, data);

        // If store size hasn't changed, we haven't added any new units
        if self.base().neighbours.len() == original_length {
            return;
        }

        self.base_mut().sort_neighbours();
        self.truncate_neighbours();
    }
}
