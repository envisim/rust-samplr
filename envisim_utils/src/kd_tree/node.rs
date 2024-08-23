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

use super::searcher::TreeSearcher;
use super::split_methods::{midpoint_slide, FindSplit};
use crate::Matrix;
use std::fmt;
use std::num::NonZeroUsize;
use thiserror::Error;

pub struct TreeBuilder<'a> {
    data: &'a Matrix<'a>,
    bucket_size: NonZeroUsize,
    split_method: FindSplit,
}

impl<'a> TreeBuilder<'a> {
    #[inline]
    pub fn new(data: &'a Matrix) -> TreeBuilder<'a> {
        Self {
            data,
            bucket_size: unsafe { NonZeroUsize::new_unchecked(40) },
            split_method: midpoint_slide,
        }
    }
    #[inline]
    pub fn bucket_size(&mut self, bucket_size: NonZeroUsize) -> Result<&mut Self, NodeError> {
        self.bucket_size = bucket_size;
        Ok(self)
    }
    #[inline]
    pub fn try_bucket_size(&mut self, bucket_size: usize) -> Result<&mut Self, NodeError> {
        match NonZeroUsize::new(bucket_size) {
            Some(bs) => {
                self.bucket_size = bs;
                Ok(self)
            }
            None => Err(NodeError::InvalidBucketSize),
        }
    }
    #[inline]
    pub fn split_method(&mut self, split_method: FindSplit) -> Result<&mut Self, NodeError> {
        self.split_method = split_method;
        Ok(self)
    }
    /// Creates a new k-d tree of the indices in untis, given a data matrix and a splitting method.
    #[inline]
    pub fn build(&self, units: &mut [usize]) -> Result<Node<'a>, NodeError> {
        if units.iter().any(|&id| self.data.nrow() <= id) {
            return Err(NodeError::GhostUnit);
        }

        let borders = Node::borders(self.data, units);
        Ok(Node::create(self, units, borders))
    }
}

impl<'a> From<&'a Matrix<'a>> for TreeBuilder<'a> {
    #[inline]
    fn from(mat: &'a Matrix) -> Self {
        TreeBuilder::new(mat)
    }
}

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NodeError {
    #[error("{0}")]
    General(String),
    #[error("unit index must not be larger than data rows")]
    GhostUnit,
    #[error("invalid bucket size")]
    InvalidBucketSize,
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
    #[inline]
    fn unwrap_branch(&self) -> &Box<NodeBranch<'a>> {
        match self {
            NodeKind::Branch(ref branch) => branch,
            _ => panic!(),
        }
    }
    #[cfg(test)]
    #[inline]
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
    fn borders(data: &'a Matrix, units: &[usize]) -> Vec<(f64, f64)> {
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

    fn create<'b>(
        options: &'b TreeBuilder<'a>,
        units: &mut [usize],
        borders: Vec<(f64, f64)>,
    ) -> Node<'a>
    where
        'a: 'b,
    {
        let TreeBuilder {
            data,
            bucket_size,
            split_method,
        } = options;
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

        Self {
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

    #[inline]
    fn new_leaf(data: &'a Matrix, units: &mut [usize]) -> Self {
        Node {
            kind: NodeKind::Leaf(Box::new(NodeLeaf {
                units: units.to_vec(),
            })),
            data,
        }
    }

    /// Returns a reference to the data matrix
    #[inline]
    pub fn data(&self) -> &Matrix {
        self.data
    }

    /// Tries to insert a unit into the tree.
    /// Returns error if the index does not exist in the data matrix.
    /// Returns `Ok(false)` if the index already existed in the tree.
    #[inline]
    pub fn insert_unit(&mut self, id: usize) -> Result<bool, NodeError> {
        if self.data.nrow() <= id {
            return Err(NodeError::GhostUnit);
        }

        Ok(self.traverse_and_alter_unit(id, true))
    }

    /// Tries to remove a unit from the tree.
    /// Returns error if the index does not exist in the data matrix.
    /// Returns `Ok(false)` if the index did not exist in the tree.
    #[inline]
    pub fn remove_unit(&mut self, id: usize) -> Result<bool, NodeError> {
        if self.data.nrow() <= id {
            return Err(NodeError::GhostUnit);
        }

        Ok(self.traverse_and_alter_unit(id, false))
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

impl<'a> fmt::Debug for Node<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            NodeKind::Leaf(ref leaf) => leaf.fmt(f),
            NodeKind::Branch(ref branch) => branch.fmt(f),
        }
    }
}
impl<'a> fmt::Debug for NodeBranch<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Branch")
            .field("dim", &self.dimension)
            .field("value", &self.value)
            .field("l", &self.left_child)
            .field("r", &self.right_child)
            .finish()
    }
}
impl fmt::Debug for NodeLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Leaf").field(&self.units).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_new<'a>() -> Matrix<'a> {
        Matrix::new(
            &[
                0.0, 1.0, 2.0, 13.0, 14.0, //
                0.0, 10.0, 20.0, 30.0, 40.0, //
            ],
            5,
        )
    }

    #[test]
    fn new_midpoint_slide() -> Result<(), NodeError> {
        let m = matrix_new();
        let t = TreeBuilder::new(&m)
            .try_bucket_size(2)?
            .build(&mut [0, 1, 2, 3])?;

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
    fn insert_unit() -> Result<(), NodeError> {
        let m = matrix_new();
        let mut t = TreeBuilder::new(&m)
            .try_bucket_size(2)?
            .build(&mut [0, 1, 2, 3])?;

        assert_eq!(t.insert_unit(4).unwrap(), true);
        assert!(t
            .kind
            .unwrap_branch()
            .right_child
            .kind
            .unwrap_leaf()
            .units
            .contains(&4));
        assert_eq!(t.insert_unit(4).unwrap(), false);

        assert_eq!(t.remove_unit(1).unwrap(), true);
        assert!(!t
            .kind
            .unwrap_branch()
            .left_child
            .kind
            .unwrap_leaf()
            .units
            .contains(&1));
        assert_eq!(t.remove_unit(1).unwrap(), false);

        assert!(t.insert_unit(10).is_err());
        assert!(t.remove_unit(10).is_err());

        Ok(())
    }
}
