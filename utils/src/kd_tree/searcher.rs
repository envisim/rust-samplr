use super::Node;
use crate::{
    matrix::{OperateMatrix, RefMatrix},
    probability::Probabilities,
};
use std::cmp::Ordering;

pub(super) trait TreeSearcher {
    fn unit(&self) -> &[f64];
    fn max_distance(&self) -> Option<f64>;
    fn is_satisfied(&self) -> bool;
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &RefMatrix);
}

pub struct Searcher {
    unit_id: usize,
    unit: Vec<f64>,
    neighbours: Vec<usize>,
    distances: Vec<f64>,
    n_neighbours: usize,
}

impl TreeSearcher for Searcher {
    fn unit(&self) -> &[f64] {
        &self.unit
    }
    fn max_distance(&self) -> Option<f64> {
        match self.neighbours.last() {
            Some(i) => Some(self.distances[*i]),
            _ => None,
        }
    }
    fn is_satisfied(&self) -> bool {
        self.neighbours.len() >= self.n_neighbours
    }
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &RefMatrix) {
        if ids.len() == 0 {
            return;
        }

        if self.n_neighbours == 1 {
            self.add_neighbours_from_node_1(ids, data);
            return;
        }

        let original_length = self.neighbours.len();
        self.add_neighbours_from_node(ids, data);

        // If store size hasn't changed, we haven't added any new units
        if self.neighbours.len() == original_length {
            return;
        }

        self.sort_neighbours();
        self.truncate_neighbours();
    }
}

impl Searcher {
    #[inline]
    pub fn new(node: &Node, n_neighbours: usize) -> Self {
        assert!(n_neighbours >= 1, "n_neighbours should be positive");
        let data = node.data();

        Self {
            unit_id: usize::MAX,
            unit: vec![0.0f64; data.ncol()],
            neighbours: Vec::<usize>::with_capacity(data.nrow()),
            distances: vec![0.0f64; data.nrow()],
            n_neighbours: n_neighbours,
        }
    }
    #[inline]
    pub fn find_neighbours(&mut self, node: &Node, unit: &[f64]) -> &mut Self {
        self.set_unit_from_iter(unit.iter(), usize::MAX);
        node.find_neighbours(self);
        self
    }
    #[inline]
    pub fn find_neighbours_of_id(&mut self, node: &Node, idx: usize) -> &mut Self {
        self.set_unit_from_iter(node.data().into_row_iter(idx), idx);
        node.find_neighbours(self);
        self
    }
    #[inline]
    pub fn find_neighbours_of_iter<'a, I>(&mut self, node: &Node, iter: I) -> &mut Self
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        self.set_unit_from_iter(iter, usize::MAX);
        node.find_neighbours(self);
        self
    }
    #[inline]
    fn set_unit_from_iter<'a, I>(&mut self, iter: I, idx: usize)
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        assert!(iter.len() == self.unit.len());
        self.unit_id = idx;
        self.unit.iter_mut().zip(iter).for_each(|(a, b)| *a = *b);
        self.reset();
    }
    #[inline]
    pub fn set_n_neighbours(&mut self, n: usize) {
        assert!(n >= 1, "n_neighbours should be positive");
        self.n_neighbours = n;
    }

    #[inline]
    pub fn neighbours(&self) -> &[usize] {
        &self.neighbours
    }

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
    fn add_neighbours_from_node_1(&mut self, ids: &[usize], data: &RefMatrix) {
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
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &RefMatrix) {
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
            } else if self.neighbours.len() < self.n_neighbours {
                self.add(id, distance);
                node_max = distance;
            }
        });
    }
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
            if i >= self.n_neighbours && self.distance_k(i - 1) < self.distance_k(i) {
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
    #[inline]
    pub fn new(node: &Node) -> Self {
        Self {
            searcher: Searcher::new(node, 1),
            weights: vec![0.0f64; node.data().nrow()],
        }
    }
    #[inline]
    pub fn find_neighbours(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        unit: &[f64],
        prob: f64,
    ) -> &mut Self {
        self.searcher.set_unit_from_iter(unit.iter(), usize::MAX);
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities: probabilities,
            total_weight: 0.0,
            unit_prob: prob,
        };
        node.find_neighbours(&mut tree_searcher);
        self
    }
    #[inline]
    pub fn find_neighbours_of_id(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        idx: usize,
    ) -> &mut Self {
        self.searcher
            .set_unit_from_iter(node.data().into_row_iter(idx), idx);
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities: probabilities,
            total_weight: 0.0,
            unit_prob: probabilities[idx],
        };
        node.find_neighbours(&mut tree_searcher);
        self
    }
    #[inline]
    pub fn find_neighbours_of_iter<'a, I>(
        &mut self,
        node: &Node,
        probabilities: &Probabilities,
        iter: I,
        prob: f64,
    ) -> &mut Self
    where
        I: ExactSizeIterator<Item = &'a f64>,
    {
        self.searcher.set_unit_from_iter(iter, usize::MAX);
        let mut tree_searcher = TreeSearcherWeighted {
            searcher: self,
            probabilities: probabilities,
            total_weight: 0.0,
            unit_prob: prob,
        };
        node.find_neighbours(&mut tree_searcher);
        self
    }

    #[inline]
    pub fn neighbours(&self) -> &[usize] {
        &self.searcher.neighbours
    }

    #[inline]
    pub fn reset(&mut self) {
        self.searcher.neighbours.clear();
    }

    #[inline]
    pub fn distance_k(&self, k: usize) -> f64 {
        self.searcher.distance_k(k)
    }
    #[inline]
    pub fn weight_k(&self, k: usize) -> f64 {
        self.weights[self.searcher.neighbours[k]]
    }
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

            return Ordering::Equal;
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

    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &RefMatrix) {
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
    fn add_neighbours_from_node(&mut self, ids: &[usize], data: &RefMatrix) {
        if ids.len() == 0 {
            return;
        }

        let original_length = self.base().neighbours.len();
        self.add_neighbours_from_node(ids, data);

        // If store size hasn't changed, we haven't added any new units
        if self.base().neighbours.len() == original_length {
            return;
        }

        self.base_mut().sort_neighbours();
        self.truncate_neighbours();
    }
}
