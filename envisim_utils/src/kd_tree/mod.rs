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

//! Implementation of a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree), together with search
//! capabilities needed by the sampling and estimations methods.
//!
//! # References
//! Lisic, J. J., & Cruze, N. B. (2016).
//! Local pivotal methods for large surveys.
//! In proceedings, ICES V, Geneva Switzerland 2016.
//! In Proceedings of the Fifth International Conference on Establishment Surveys.

mod point_access;
pub mod searcher;
pub mod split_methods;

use std::num::NonZeroUsize;

pub use point_access::PointAccess;
use searcher::TreeSearcher;
use split_methods::{
    Border,
    FindSplit,
    Split,
};

pub trait TreeBuilder {
    type Data: PointAccess;
    fn data(&self) -> &Self::Data;
    fn bucket_size(&self) -> NonZeroUsize;
    fn split_method(&self) -> FindSplit<Self::Data>;
}

#[derive(Clone, Debug)]
pub struct Tree<'a, T>
where
    T: PointAccess,
{
    node: Node,
    data: &'a T,
}
#[derive(Clone, Debug)]
#[allow(clippy::exhaustive_enums)]
pub enum Node {
    Branch(Branch),
    Leaf(Leaf),
}
#[derive(Clone, Debug)]
pub struct Branch {
    split: Split,
    left: Box<Node>,
    right: Box<Node>,
}
#[derive(Clone, Debug)]
pub struct Leaf {
    units: Vec<usize>,
}

impl<'a, T> Tree<'a, T>
where
    T: PointAccess,
{
    /// A builder with a [`FindSplit`] that is able to return split unit of 0 or len will panic.
    #[inline]
    pub fn new<B>(builder: &'a B, units: &mut [usize]) -> Self
    where
        B: TreeBuilder<Data = T>,
        T: PointAccess,
    {
        let data = builder.data();
        let borders = Border::from_data_to_vec(data, units);

        Self {
            node: Node::new(builder, borders, units),
            data,
        }
    }
    #[inline]
    pub fn data(&self) -> &T { self.data }
    #[inline]
    pub fn find_leaf_of_unit(&self, unit: usize) -> Option<&Leaf> {
        let v: Box<[f64]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, find_leaf should always be Some as there can't be
        // dimension mismatch
        self.find_leaf(&v)
    }
    #[inline]
    pub fn find_leaf(&self, unit: &[f64]) -> Option<&Leaf> {
        (unit.len() == self.data.dim().get()).then(|| self.node.find_leaf(unit))
    }
    #[inline]
    pub fn iterate_leafs_by<S>(&self, searcher: &mut S) -> Option<()>
    where
        S: TreeSearcher,
    {
        if self.data.dim().get() == searcher.point().len() {
            self.node.iterate_leafs_by(self.data, searcher)
        } else {
            None
        }
    }
    #[inline]
    pub fn find_leaf_of_unit_mut(&mut self, unit: usize) -> Option<&mut Leaf> {
        let v: Box<[f64]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, it should always be Some
        self.find_leaf_mut(&v)
    }
    #[inline]
    pub fn find_leaf_mut(&mut self, unit: &[f64]) -> Option<&mut Leaf> {
        (unit.len() == self.data.dim().get()).then(|| self.node.find_leaf_mut(unit))
    }

    /// Returns None if leaf or unit cannot be found/is invalid.
    /// Returns Some(true) if unit did not already exist.
    /// Does not rebalance the tree.
    #[inline]
    pub fn insert_unit(&mut self, unit: usize) -> Option<bool> {
        self.find_leaf_of_unit_mut(unit)?.insert_unit(unit).into()
    }
    /// Returns None if leaf or unit cannot be found/is invalid.
    /// Returns Some(true) if unit existed.
    /// Does not rebalance the tree.
    #[inline]
    pub fn remove_unit(&mut self, unit: usize) -> Option<bool> {
        self.find_leaf_of_unit_mut(unit)?.remove_unit(unit).into()
    }
}

impl Node {
    fn new<B>(builder: &B, borders: Box<[Border]>, units: &mut [usize]) -> Self
    where
        B: TreeBuilder,
    {
        // If not enough units remain, a leaf should be constructed
        if units.len() <= builder.bucket_size().get() {
            return Leaf::new(units).into();
        }

        // Try to find a split, and if not possible, construct a leaf
        let Some(split) = builder.split_method()(builder.data(), &borders, units) else {
            return Leaf::new(units).into();
        };

        let mut l_borders = borders.clone();
        l_borders[split.dimension()].max = split.value();
        let mut r_borders = borders;
        r_borders[split.dimension()].min = split.value();

        let unit = split.unit();
        assert!(
            0 < unit && unit < units.len(),
            "split_method failed to find valid split without returning None"
        );

        Branch {
            split: split.into(),
            left: Self::new(builder, l_borders, &mut units[..unit]).into(),
            right: Self::new(builder, r_borders, &mut units[unit..]).into(),
        }
        .into()
    }
    fn find_leaf(&self, unit: &[f64]) -> &Leaf {
        match self {
            Self::Branch(branch) => {
                if branch.split.unit_is_left(unit) {
                    branch.left.find_leaf(unit)
                } else {
                    branch.right.find_leaf(unit)
                }
            }
            Self::Leaf(leaf) => leaf,
        }
    }
    fn find_leaf_mut(&mut self, unit: &[f64]) -> &mut Leaf {
        match self {
            Self::Branch(branch) => {
                if branch.split.unit_is_left(unit) {
                    branch.left.find_leaf_mut(unit)
                } else {
                    branch.right.find_leaf_mut(unit)
                }
            }
            Self::Leaf(leaf) => leaf,
        }
    }
    #[inline]
    fn iterate_leafs_by<T, S>(&self, data: &T, searcher: &mut S) -> Option<()>
    where
        T: PointAccess,
        S: TreeSearcher,
    {
        match self {
            Self::Branch(branch) => {
                let (first, second) = if branch.split.unit_is_left(searcher.point()) {
                    (&branch.left, &branch.right)
                } else {
                    (&branch.right, &branch.left)
                };

                first.iterate_leafs_by(data, searcher)?;

                let distance = branch.split.unit_distance(searcher.point());
                if !searcher.is_satisfied(distance) {
                    second.iterate_leafs_by(data, searcher)?;
                }
            }
            Self::Leaf(leaf) => {
                if !leaf.units.is_empty() {
                    searcher.visit_leaf(data, &leaf.units);
                }
            }
        }
        ().into()
    }
}
impl Branch {
    #[inline]
    pub fn split(&self) -> &Split { &self.split }
    #[inline]
    pub fn left(&self) -> &Node { &self.left }
    #[inline]
    pub fn right(&self) -> &Node { &self.right }
}
impl Leaf {
    #[inline]
    fn new(units: &[usize]) -> Self {
        Self {
            units: units.to_vec(),
        }
    }
    #[inline]
    pub fn units(&self) -> &[usize] { &self.units }
    #[inline]
    pub fn contains_unit(&self, unit: usize) -> bool { self.units.contains(&unit) }
    #[inline]
    fn insert_unit(&mut self, unit: usize) -> bool {
        if !self.contains_unit(unit) {
            self.units.push(unit);
            return true;
        }
        false
    }
    #[inline]
    fn remove_unit(&mut self, unit: usize) -> bool {
        match self.units.iter().position(|&id| id == unit) {
            Some(idx) => {
                self.units.swap_remove(idx);
                true
            }
            _ => false,
        }
    }
}

impl From<Branch> for Node {
    fn from(branch: Branch) -> Self { Node::Branch(branch) }
}
impl From<Leaf> for Node {
    fn from(leaf: Leaf) -> Self { Node::Leaf(leaf) }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::{
        PointAccess,
        Tree,
    };
    use crate::matrix::Matrix;
    use crate::sampling_options::SpreadingOptions;

    fn test_builder<'a>(data: Matrix<'a>) -> SpreadingOptions<'a> {
        SpreadingOptions::new(data).unwrap()
    }

    fn mat_from(data: Vec<f64>, rows: usize) -> Matrix<'static> {
        Matrix::from_vec(data, NonZeroUsize::new(rows).unwrap()).unwrap()
    }

    // Grid of 9 2D points: (0,0), (0,1), (0,2), (1,0), ..., (2,2)
    fn grid_3x3() -> Matrix<'static> {
        let data = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, // dim 0
            0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, // dim 1
        ];
        mat_from(data, 9)
    }

    #[test]
    fn tree_with_bucket_larger_than_population_is_single_leaf() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(100).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let tree = Tree::new(&builder, &mut units);

        // Every unit should resolve to the same leaf
        for id in 0..9 {
            let leaf = tree.find_leaf_of_unit(id).unwrap();
            assert_eq!(leaf.units().len(), 9);
            assert!(leaf.contains_unit(id));
        }
    }

    #[test]
    fn tree_with_small_bucket_partitions_units() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let tree = Tree::new(&builder, &mut units);

        // Every unit must be findable in exactly one leaf
        for id in 0..9 {
            let leaf = tree.find_leaf_of_unit(id).unwrap();
            assert!(leaf.units().len() <= 2);
            assert!(leaf.contains_unit(id));
        }
    }

    #[test]
    fn find_leaf_returns_none_for_wrong_dim() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let tree = Tree::new(&builder, &mut units);

        // tree is 2D; probing with a 3D point must fail
        assert!(tree.find_leaf(&[0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn find_leaf_of_unit_returns_none_for_invalid_id() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let tree = Tree::new(&builder, &mut units);

        assert!(tree.find_leaf_of_unit(100).is_none());
    }

    #[test]
    fn insert_unit_into_correct_leaf() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..8).collect(); // insert id 8 later
        let mut tree = Tree::new(&builder, &mut units);

        // id 8 initially not in any leaf
        let leaf_before = tree.find_leaf_of_unit(8).unwrap();
        assert!(!leaf_before.contains_unit(8));

        assert_eq!(
            tree.insert_unit(8),
            Some(true),
            "insertion of a new unit should return true"
        );

        let leaf_after = tree.find_leaf_of_unit(8).unwrap();
        assert!(leaf_after.contains_unit(8));

        assert_eq!(
            tree.insert_unit(8),
            Some(false),
            "insertion of a new unit should return false"
        );

        assert_eq!(
            tree.insert_unit(999),
            None,
            "insertion of an invalid id should return None"
        );
    }

    #[test]
    fn remove_unit_that_exists() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let mut tree = Tree::new(&builder, &mut units);

        let leaf_before = tree.find_leaf_of_unit(4).unwrap();
        assert!(leaf_before.contains_unit(4));

        assert_eq!(
            tree.remove_unit(4),
            Some(true),
            "removal of an existing unit should return true"
        );

        let leaf_after = tree.find_leaf_of_unit(4).unwrap();
        assert!(!leaf_after.contains_unit(4));

        assert_eq!(
            tree.remove_unit(4),
            Some(false),
            "removal of a non-existing unit should return false"
        );

        assert_eq!(
            tree.remove_unit(400),
            None,
            "removal of an invalid unit should return None"
        );
    }

    #[test]
    fn tree_data_returns_backing_pointaccess() {
        let m = grid_3x3();
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..9).collect();
        let tree = Tree::new(&builder, &mut units);

        assert_eq!(tree.data().dim().get(), 2);
        assert_eq!(tree.data().coord(4, 0), 1.0);
        assert_eq!(tree.data().coord(4, 1), 1.0);
    }

    #[test]
    fn tree_handles_single_point() {
        let m = mat_from(vec![1.0, 2.0], 1);
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = vec![0];
        let tree = Tree::new(&builder, &mut units);

        let leaf = tree.find_leaf_of_unit(0).unwrap();
        assert!(leaf.contains_unit(0));
        assert_eq!(leaf.units().len(), 1);
    }

    #[test]
    fn tree_handles_collocated_points_as_leaf() {
        // All 4 points identical -> midpoint_slide returns None -> single leaf
        let m = mat_from(vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], 4);
        let builder = test_builder(m).set_bucket_size(2).unwrap();
        let mut units: Vec<usize> = (0..4).collect();
        let tree = Tree::new(&builder, &mut units);

        // Even though bucket_size < n, no split is possible, so we must have
        // one leaf with all 4 units.
        let leaf = tree.find_leaf_of_unit(0).unwrap();
        assert_eq!(leaf.units().len(), 4);
    }
}
