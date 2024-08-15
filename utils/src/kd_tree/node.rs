use super::searcher::TreeSearcher;
use super::split_methods::{midpoint_slide, FindSplit};
use crate::indices::Indices;
use crate::matrix::{OperateMatrix, RefMatrix};
use std::num::NonZeroUsize;
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum NodeError {
    #[error("{0}")]
    General(String),
    #[error("unit index must not be larger than data rows")]
    GhostUnit,
}

struct NodeBranch<'a> {
    dimension: usize,
    value: f64,
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

pub struct Node<'a> {
    kind: NodeKind<'a>,

    // Common
    data: &'a RefMatrix<'a>,
    min_border: Vec<f64>,
    max_border: Vec<f64>,
}

impl<'a> Node<'a> {
    fn create(
        find_split: FindSplit,
        bucket_size: NonZeroUsize,
        data: &'a RefMatrix,
        units: &mut [usize],
    ) -> Self {
        let mut min_border = Vec::<f64>::with_capacity(data.ncol());
        let mut max_border = Vec::<f64>::with_capacity(data.ncol());

        let mut node_size_is_zero = true;

        for k in 0usize..data.ncol() {
            let (min, max) =
                units
                    .iter()
                    .map(|&id| data[(id, k)])
                    .fold((f64::MAX, 0.0f64), |(min, max), v| {
                        let nmin = min.min(v);
                        let nmax = max.max(v);
                        (nmin, nmax)
                    });

            min_border.push(min);
            max_border.push(max);

            if node_size_is_zero && max - min > f64::EPSILON {
                node_size_is_zero = false;
            }
        }

        if node_size_is_zero || units.len() <= bucket_size.get() {
            return Node::new_leaf(data, units, min_border, max_border);
        }

        let split = match find_split(&min_border, &max_border, data, &mut *units) {
            Some(s) => s,
            None => return Self::new_leaf(data, units, min_border, max_border),
        };

        assert!(split.dimension < data.ncol());

        Self {
            kind: NodeKind::Branch(Box::new(NodeBranch {
                dimension: split.dimension,
                value: split.value,
                left_child: Box::new(Self::create(
                    find_split,
                    bucket_size,
                    data,
                    &mut units[..split.unit],
                )),
                right_child: Box::new(Self::create(
                    find_split,
                    bucket_size,
                    data,
                    &mut units[split.unit..],
                )),
            })),
            data,
            min_border,
            max_border,
        }
    }

    #[inline]
    fn new_leaf(
        data: &'a RefMatrix,
        units: &mut [usize],
        min_border: Vec<f64>,
        max_border: Vec<f64>,
    ) -> Self {
        Node {
            kind: NodeKind::Leaf(Box::new(NodeLeaf {
                units: units.to_vec(),
            })),
            data,
            min_border,
            max_border,
        }
    }

    #[inline]
    pub fn new(
        find_split: FindSplit,
        bucket_size: NonZeroUsize,
        data: &'a RefMatrix,
        units: &mut [usize],
    ) -> Result<Self, NodeError> {
        if units.iter().any(|&id| data.nrow() <= id) {
            return Err(NodeError::GhostUnit);
        }

        Ok(Self::create(find_split, bucket_size, data, units))
    }

    #[inline]
    pub fn new_midpoint_slide(
        bucket_size: NonZeroUsize,
        data: &'a RefMatrix,
        units: &mut [usize],
    ) -> Result<Self, NodeError> {
        Self::new(midpoint_slide, bucket_size, data, units)
    }

    #[inline]
    pub fn new_from_indices(
        find_split: FindSplit,
        bucket_size: NonZeroUsize,
        data: &'a RefMatrix,
        indices: &Indices,
    ) -> Result<Self, NodeError> {
        Self::new(find_split, bucket_size, data, &mut indices.list().to_vec())
    }

    #[inline]
    pub fn data(&self) -> &RefMatrix {
        self.data
    }

    #[inline]
    pub fn insert_unit(&mut self, id: usize) -> Result<bool, NodeError> {
        if self.data.nrow() <= id {
            return Err(NodeError::GhostUnit);
        }

        Ok(self.traverse_and_alter_unit(id, true))
    }

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

                if distance <= 0.0 {
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

    #[inline]
    fn distance_to_box_in_dimension(&self, dimension: usize, value: f64) -> f64 {
        let min = self.min_border[dimension];
        let max = self.max_border[dimension];

        if value <= min {
            min - value
        } else if value <= max {
            0.0
        } else {
            value - max
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
                let dist = unit_value - branch.value;

                let (first_node, second_node) = if dist <= 0.0 {
                    (&branch.left_child, &branch.right_child)
                } else {
                    (&branch.right_child, &branch.left_child)
                };

                if !searcher.is_satisfied()
                    || first_node
                        .distance_to_box_in_dimension(branch.dimension, unit_value)
                        .powi(2)
                        <= searcher.max_distance().unwrap_or(f64::INFINITY)
                {
                    first_node.find_neighbours(searcher);
                }

                if !searcher.is_satisfied()
                    || second_node
                        .distance_to_box_in_dimension(branch.dimension, unit_value)
                        .powi(2)
                        <= searcher.max_distance().unwrap_or(f64::INFINITY)
                {
                    second_node.find_neighbours(searcher);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;

    const DATA_4_2: [f64; 10] = [
        0.0, 1.0, 2.0, 13.0, 14.0, //
        0.0, 10.0, 20.0, 30.0, 40.0, //
    ];

    fn matrix_new<'a>() -> RefMatrix<'a> {
        RefMatrix::new(&DATA_4_2, 5)
    }

    #[test]
    fn new_midpoint_slide() {
        let m = matrix_new();
        let t = Node::new_midpoint_slide(NONZERO_2, &m, &mut vec![0, 1, 2, 3]).unwrap();

        let branch = t.kind.unwrap_branch();
        assert_eq!(branch.dimension, 1);
        assert!((10.0..20.0).contains(&branch.value));

        let l = &branch.left_child;
        assert_eq!(l.min_border, vec![0.0, 0.0]);
        assert_eq!(l.max_border, vec![1.0, 10.0]);
        let l = &l.kind;
        assert!(l.unwrap_leaf().units.contains(&0));
        assert!(l.unwrap_leaf().units.contains(&1));
        assert!(!l.unwrap_leaf().units.contains(&2));
        assert!(!l.unwrap_leaf().units.contains(&3));
        assert!(!l.unwrap_leaf().units.contains(&4));

        let r = &branch.right_child;
        assert_eq!(r.min_border, vec![2.0, 20.0]);
        assert_eq!(r.max_border, vec![13.0, 30.0]);
        let r = &r.kind;
        assert!(!r.unwrap_leaf().units.contains(&0));
        assert!(!r.unwrap_leaf().units.contains(&1));
        assert!(r.unwrap_leaf().units.contains(&2));
        assert!(r.unwrap_leaf().units.contains(&3));
        assert!(!r.unwrap_leaf().units.contains(&4));
    }

    #[test]
    fn insert_unit() {
        let m = matrix_new();
        let mut t = Node::new_midpoint_slide(NONZERO_2, &m, &mut vec![0, 1, 2, 3]).unwrap();

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
    }

    #[test]
    fn distance_to_box_in_dimension() {
        let m = matrix_new();
        let t = Node::new_midpoint_slide(NONZERO_2, &m, &mut vec![0, 1, 2, 3]).unwrap();

        assert_eq!(t.distance_to_box_in_dimension(0, 5.0), 0.0);
        assert_eq!(t.distance_to_box_in_dimension(0, -5.0), 5.0);
        assert_eq!(t.distance_to_box_in_dimension(0, 19.0), 6.0);
        assert_eq!(t.distance_to_box_in_dimension(1, -10.0), 10.0);
        assert_eq!(
            t.kind
                .unwrap_branch()
                .right_child
                .distance_to_box_in_dimension(1, -10.0),
            30.0
        );
    }
}
