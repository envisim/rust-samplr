use super::{searcher::TreeSearcher, split_methods::FindSplit};
use crate::{
    indices::Indices,
    matrix::{OperateMatrix, RefMatrix},
};

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
    pub fn new(
        find_split: FindSplit,
        bucket_size: usize,
        data: &'a RefMatrix,
        units: &mut [usize],
    ) -> Self {
        assert!(bucket_size > 0);

        let mut min_border = Vec::<f64>::with_capacity(data.ncol());
        let mut max_border = Vec::<f64>::with_capacity(data.ncol());

        let mut node_size_is_zero = true;

        for k in 0usize..data.ncol() {
            let mut min = data[(units[0], k)];
            let mut max = min;

            units.iter().for_each(|&id| {
                let val: f64 = data[(id, k)];
                if val < min {
                    min = val;
                } else if val > max {
                    max = val;
                }
            });

            if node_size_is_zero && max - min > f64::EPSILON {
                node_size_is_zero = false;
            }

            min_border.push(min);
            max_border.push(max);
        }

        if node_size_is_zero || units.len() <= bucket_size {
            return Node::new_leaf(data, units, min_border, max_border);
        }

        let (split_unit, split_dimension, split_value) =
            find_split(&min_border, &max_border, data, &mut *units);

        if split_unit == 0 || split_unit >= units.len() {
            return Node::new_leaf(data, units, min_border, max_border);
        }

        assert!(split_dimension < data.ncol());

        Node {
            kind: NodeKind::Branch(Box::new(NodeBranch {
                dimension: split_dimension,
                value: split_value,
                left_child: Box::new(Node::new(
                    find_split,
                    bucket_size,
                    data,
                    &mut units[..split_unit],
                )),
                right_child: Box::new(Node::new(
                    find_split,
                    bucket_size,
                    data,
                    &mut units[split_unit..],
                )),
            })),
            data: data,
            min_border: min_border,
            max_border: max_border,
        }
    }

    #[inline]
    pub fn new_from_indices(
        find_split: FindSplit,
        bucket_size: usize,
        data: &'a RefMatrix,
        indices: &Indices,
    ) -> Self {
        Node::new(find_split, bucket_size, data, &mut indices.list().to_vec())
    }

    pub fn new_leaf(
        data: &'a RefMatrix,
        units: &mut [usize],
        min_border: Vec<f64>,
        max_border: Vec<f64>,
    ) -> Self {
        Node {
            kind: NodeKind::Leaf(Box::new(NodeLeaf {
                units: units.to_vec(),
            })),
            data: data,
            min_border: min_border,
            max_border: max_border,
        }
    }

    #[inline]
    pub fn data(&self) -> &RefMatrix {
        self.data
    }

    pub fn remove_unit(&mut self, id: usize) -> bool {
        match self.kind {
            NodeKind::Branch(ref mut branch) => {
                let distance = self.data[(id, branch.dimension)] - branch.value;

                if distance <= 0.0 {
                    branch.left_child.remove_unit(id)
                } else {
                    branch.right_child.remove_unit(id)
                }
            }
            NodeKind::Leaf(ref mut leaf) => {
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

    #[inline]
    fn distance_to_box_in_dimension(&self, dimension: usize, value: f64) -> f64 {
        unsafe {
            let min = *self.min_border.get_unchecked(dimension);
            let max = *self.max_border.get_unchecked(dimension);

            if value <= min {
                return min - value;
            } else if value <= max {
                return 0.0;
            } else {
                return value - max;
            }
        }
    }

    pub(super) fn find_neighbours<S>(&self, searcher: &mut S)
    where
        S: TreeSearcher,
    {
        match self.kind {
            NodeKind::Leaf(ref leaf) => {
                searcher.add_neighbours_from_node(&leaf.units, self.data);
                return;
            }

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
    use super::super::searcher::Searcher;
    use super::*;
    use crate::{kd_tree::split_methods::midpoint_slide, matrix::RefMatrix};

    const DATA_10_2: [f64; 20] = [
        0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
        0.66079779, 0.62911404, 0.06178627, //
        0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
        0.3800352, 0.7774452,
    ];

    fn data_10_2<'a>() -> (RefMatrix<'a>, [f64; 10], Vec<usize>) {
        (
            RefMatrix::new(&DATA_10_2, 10),
            [0.2f64; 10],
            (0..10).collect::<Vec<usize>>(),
        )
    }

    const DATA_9_1: [f64; 9] = [2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2];

    fn data_9_1<'a>() -> (RefMatrix<'a>, [f64; 9], Vec<usize>) {
        (
            RefMatrix::new(&DATA_9_1, 9),
            [0.1f64; 9],
            (0..9).collect::<Vec<usize>>(),
        )
    }

    #[test]
    fn find() {
        let (data, _prob, mut indices) = data_10_2();
        let root = Node::new(midpoint_slide, 2, &data, &mut indices);
        let mut searcher = Searcher::new(&root, 1);

        let closest_neighbour: [usize; 10] = [1, 0, 8, 5, 9, 3, 5, 2, 3, 4];
        let mut result: [usize; 10] = [0; 10];

        for i in 0usize..10 {
            result[i] = searcher.find_neighbours_of_id(&root, i).neighbours()[0];
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn find_and_remove() {
        let (data, _prob, mut indices) = data_10_2();
        let mut root = Node::new(midpoint_slide, 2, &data, &mut indices);
        let mut searcher = Searcher::new(&root, 1);

        let closest_neighbour: [usize; 9] = [1, 8, 8, 5, 9, 6, 7, 8, 9];
        let mut result: [usize; 9] = [0; 9];

        for i in 0usize..9 {
            result[i] = searcher.find_neighbours_of_id(&root, i).neighbours()[0];
            root.remove_unit(i);
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn node_splits() {
        let (data, _prob, mut indices) = data_9_1();
        let n0 = Node::new(midpoint_slide, 3, &data, &mut indices);

        let n0s = n0.kind.unwrap_branch();

        assert_eq!(n0s.dimension, 0);
        assert!(n0s.value >= 2.2 && n0s.value <= 10.0);

        let n00 = &n0s.left_child;
        let n01 = &n0s.right_child;

        let n00s = n00.kind.unwrap_branch();

        assert_eq!(n00s.dimension, 0);
        assert!(n00s.value >= 1.2 && n00s.value <= 2.0);

        let n000 = &n00s.left_child;
        let n001 = &n00s.right_child;

        let mut n01u = n01.kind.unwrap_leaf().units.to_vec();
        let mut n000u = n000.kind.unwrap_leaf().units.to_vec();
        let mut n001u = n001.kind.unwrap_leaf().units.to_vec();
        n01u.sort();
        n000u.sort();
        n001u.sort();

        assert_eq!(*n000u, vec![6usize, 7, 8]);
        assert_eq!(*n001u, vec![0usize, 1, 2]);
        assert_eq!(*n01u, vec![3usize, 4, 5]);
    }
}
