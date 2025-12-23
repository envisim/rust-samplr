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

use std::num::NonZeroUsize;

use super::FindSplit;
use super::searcher::TreeSearcher;
use crate::matrix::Matrix;
use crate::sampling_options::SpreadingOptions;

pub trait TreeBuilder<'a> {
    fn data(&self) -> &Matrix<'a>;
    fn bucket_size(&self) -> NonZeroUsize;
    fn split_method(&self) -> FindSplit;
    /// Creates a new k-d tree of the indices in untis, given a data matrix and a splitting method.
    fn build(&'a self, units: &mut [usize]) -> Option<Node<'a>>;
}

impl<'a> TreeBuilder<'a> for SpreadingOptions<'a> {
    fn data(&self) -> &Matrix<'a> { self.data() }
    fn bucket_size(&self) -> NonZeroUsize { self.bucket_size() }
    fn split_method(&self) -> FindSplit { self.split_method() }
    fn build(&'a self, units: &mut [usize]) -> Option<Node<'a>> {
        let population_size = self.data().nrow();
        if units.iter().any(|&id| id >= population_size) {
            return None;
        }

        let borders = Node::borders(self.data(), units);
        Some(Node::create(self, units, borders))
    }
}

struct NodeBranch<'a> {
    dimension: usize,
    value: f64,
    leq: bool,
    left_child: Box<Node<'a>>,
    right_child: Box<Node<'a>>,
}

struct NodeLeaf {
    units: Vec<usize>,
}

enum NodeKind<'a> {
    Branch(Box<NodeBranch<'a>>),
    Leaf(Box<NodeLeaf>),
}

impl<'a> NodeKind<'a> {
    #[cfg(test)]
    fn unwrap_branch(&self) -> &Box<NodeBranch<'a>> {
        match self {
            NodeKind::Branch(ref branch) => branch,
            _ => panic!(),
        }
    }
    #[cfg(test)]
    fn unwrap_leaf(&self) -> &Box<NodeLeaf> {
        match self {
            NodeKind::Leaf(ref leaf) => leaf,
            _ => panic!(),
        }
    }
}

/// A struct containing a k-d tree
pub struct Node<'a> {
    kind: NodeKind<'a>,

    // Common
    data: &'a Matrix<'a>,
}

impl<'a> Node<'a> {
    fn borders(data: &Matrix, units: &[usize]) -> Vec<(f64, f64)> {
        let mut b = Vec::<(f64, f64)>::with_capacity(data.ncol());

        for k in 0usize..data.ncol() {
            b.push(units.iter().map(|&id| data[(id, k)]).fold(
                (f64::MAX, f64::MIN),
                |(min, max), v| {
                    let nmin = min.min(v);
                    let nmax = max.max(v);
                    (nmin, nmax)
                },
            ));
        }

        b
    }

    fn create(
        options: &'a impl TreeBuilder<'a>,
        units: &mut [usize],
        borders: Vec<(f64, f64)>,
    ) -> Node<'a> {
        let data = options.data();
        let bucket_size = options.bucket_size();
        let split_method = options.split_method();
        if units.len() <= bucket_size.get() {
            return Node::new_leaf(data, units);
        }

        let split = match split_method(&borders, data, &mut *units) {
            Some(s) => s,
            None => return Self::new_leaf(data, units),
        };

        assert!(split.dimension < data.ncol());

        let mut l_borders = borders.to_vec();
        let mut r_borders = borders.to_vec();
        l_borders[split.dimension].1 = split.value;
        r_borders[split.dimension].0 = split.value;

        Node {
            kind: NodeKind::Branch(Box::new(NodeBranch {
                dimension: split.dimension,
                value: split.value,
                leq: split.leq,
                left_child: Box::new(Self::create(options, &mut units[..split.unit], l_borders)),
                right_child: Box::new(Self::create(options, &mut units[split.unit..], r_borders)),
            })),
            data,
        }
    }

    fn new_leaf(data: &'a Matrix<'a>, units: &mut [usize]) -> Node<'a> {
        Node {
            kind: NodeKind::Leaf(Box::new(NodeLeaf {
                units: units.to_vec(),
            })),
            data,
        }
    }

    /// Returns a reference to the data matrix
    pub fn data(&'a self) -> &'a Matrix<'a> { self.data }

    /// Tries to insert a unit into the tree.
    /// Returns error if the index does not exist in the data matrix.
    /// Returns `Ok(false)` if the index already existed in the tree.
    pub fn insert_unit(&mut self, id: usize) -> Option<bool> {
        if id >= self.data.nrow() {
            return None;
        }
        Some(self.traverse_and_alter_unit(id, true))
    }

    /// Tries to remove a unit from the tree.
    /// Returns error if the index does not exist in the data matrix.
    /// Returns `Ok(false)` if the index did not exist in the tree.
    pub fn remove_unit(&mut self, id: usize) -> Option<bool> {
        if id >= self.data.nrow() {
            return None;
        }
        Some(self.traverse_and_alter_unit(id, false))
    }

    fn traverse_and_alter_unit(&mut self, id: usize, insert: bool) -> bool {
        match self.kind {
            NodeKind::Branch(ref mut branch) => {
                let distance = self.data[(id, branch.dimension)] - branch.value;

                if distance < 0.0 || (branch.leq && distance == 0.0) {
                    branch.left_child.traverse_and_alter_unit(id, insert)
                } else {
                    branch.right_child.traverse_and_alter_unit(id, insert)
                }
            }
            NodeKind::Leaf(ref mut leaf) => {
                if insert {
                    if leaf.units.contains(&id) {
                        false
                    } else {
                        leaf.units.push(id);
                        true
                    }
                } else {
                    let vec_k = leaf.units.iter().position(|&p| p == id);

                    match vec_k {
                        Some(k) => {
                            leaf.units.swap_remove(k);
                            true
                        }
                        _ => false,
                    }
                }
            }
        }
    }

    pub(super) fn find_neighbours<S>(&self, searcher: &mut S)
    where
        S: TreeSearcher,
    {
        match self.kind {
            NodeKind::Leaf(ref leaf) => searcher.add_neighbours_from_node(&leaf.units, self.data),

            NodeKind::Branch(ref branch) => {
                let unit_value = searcher.unit()[branch.dimension];
                let distance = unit_value - branch.value;

                let (first_node, second_node) = if distance < 0.0 || (branch.leq && distance == 0.0)
                {
                    (&branch.left_child, &branch.right_child)
                } else {
                    (&branch.right_child, &branch.left_child)
                };

                first_node.find_neighbours(searcher);

                if !searcher.is_satisfied()
                    || distance.powi(2) <= searcher.max_distance().unwrap_or(f64::INFINITY)
                {
                    second_node.find_neighbours(searcher);
                }
            }
        }
    }
}

impl<'a> std::fmt::Debug for Node<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            NodeKind::Leaf(ref leaf) => leaf.fmt(f),
            NodeKind::Branch(ref branch) => branch.fmt(f),
        }
    }
}
impl<'a> std::fmt::Debug for NodeBranch<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Branch")
            .field("dim", &self.dimension)
            .field("value", &self.value)
            .field("l", &self.left_child)
            .field("r", &self.right_child)
            .finish()
    }
}
impl std::fmt::Debug for NodeLeaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Leaf").field(&self.units).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_options::SamplingOptionsError;

    static MATRIX_DATA: [f64; 10] = [
        0.0, 1.0, 2.0, 13.0, 14.0, //
        0.0, 10.0, 20.0, 30.0, 40.0, //
    ];
    fn matrix_new<'a>() -> Matrix<'a> { Matrix::new(&MATRIX_DATA, 5).unwrap() }

    #[test]
    fn new_midpoint_slide() -> Result<(), SamplingOptionsError> {
        let m = matrix_new();
        let opts = SpreadingOptions::new(m).set_bucket_size(2)?;
        let t = opts.build(&mut [0, 1, 2, 3]).unwrap();
        println!("{:?}", t);

        let branch = t.kind.unwrap_branch();
        assert_eq!(branch.dimension, 1);
        assert!((10.0..20.0).contains(&branch.value));

        let l = &branch.left_child;
        let l = &l.kind;
        assert!(l.unwrap_leaf().units.contains(&0));
        assert!(l.unwrap_leaf().units.contains(&1));
        assert!(!l.unwrap_leaf().units.contains(&2));
        assert!(!l.unwrap_leaf().units.contains(&3));
        assert!(!l.unwrap_leaf().units.contains(&4));

        let r = &branch.right_child;
        let r = &r.kind;
        assert!(!r.unwrap_leaf().units.contains(&0));
        assert!(!r.unwrap_leaf().units.contains(&1));
        assert!(r.unwrap_leaf().units.contains(&2));
        assert!(r.unwrap_leaf().units.contains(&3));
        assert!(!r.unwrap_leaf().units.contains(&4));

        Ok(())
    }

    #[test]
    fn insert_unit() -> Result<(), SamplingOptionsError> {
        let m = matrix_new();
        let opts = SpreadingOptions::new(m).set_bucket_size(2)?;
        let mut t = opts.build(&mut [0, 1, 2, 3]).unwrap();

        assert_eq!(t.insert_unit(4).unwrap(), true);
        assert!(
            t.kind
                .unwrap_branch()
                .right_child
                .kind
                .unwrap_leaf()
                .units
                .contains(&4)
        );
        assert_eq!(t.insert_unit(4).unwrap(), false);

        assert_eq!(t.remove_unit(1).unwrap(), true);
        assert!(
            !t.kind
                .unwrap_branch()
                .left_child
                .kind
                .unwrap_leaf()
                .units
                .contains(&1)
        );
        assert_eq!(t.remove_unit(1).unwrap(), false);

        assert!(t.insert_unit(10).is_none());
        assert!(t.remove_unit(10).is_none());

        Ok(())
    }
}
