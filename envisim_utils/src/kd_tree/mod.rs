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

pub mod searcher;
pub mod split_methods;

use std::num::NonZeroUsize;

use searcher::TreeSearcher;
use split_methods::{
    FindSplit,
    Split,
};

pub use crate::number_traits::Number;
pub use crate::spatial::PointSet;

pub trait TreeConfig<N>
where
    N: Number,
{
    type Data: PointSet<N>;
    type Split: FindSplit<N, Self::Data>;
    fn data(&self) -> &Self::Data;
    fn bucket_size(&self) -> NonZeroUsize;
    fn split_method(&self, units: &[usize]) -> Self::Split;
}

#[derive(Clone, Debug)]
pub struct Tree<'b, N, T> {
    node: Node<N>,
    data: &'b T,
}
#[derive(Clone, Debug)]
#[allow(clippy::exhaustive_enums)]
pub enum Node<N> {
    Branch(Branch<N>),
    Leaf(Leaf),
}
#[derive(Clone, Debug)]
pub struct Branch<N> {
    split: Split<N>,
    left: Box<Node<N>>,
    right: Box<Node<N>>,
}
#[derive(Clone, Debug)]
pub struct Leaf {
    units: Vec<usize>,
}

impl<'b, N, T> Tree<'b, N, T>
where
    N: Number,
    T: PointSet<N>,
{
    /// A builder with a [`FindSplit`] that is able to return split unit of 0 or len will panic.
    #[inline]
    pub fn new<C, S>(config: &'b C, units: &mut [usize]) -> Self
    where
        C: TreeConfig<N, Data = T, Split = S>,
        S: FindSplit<N, T>,
    {
        let data = config.data();
        let borders = config.split_method(units);
        let node = Node::new(config, borders, units);

        Self { node, data }
    }
    #[inline]
    pub fn data(&self) -> &T { self.data }
    #[inline]
    pub fn find_leaf_of_unit(&self, unit: usize) -> Option<&Leaf> {
        let v: Box<[N]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, find_leaf should always be Some as there can't be
        // dimension mismatch
        self.find_leaf(&v)
    }
    #[inline]
    pub fn find_leaf(&self, unit: &[N]) -> Option<&Leaf> {
        (unit.len() == self.data.dim().get()).then(|| self.node.find_leaf(unit))
    }
    #[inline]
    pub fn iterate_leafs_by<S>(&self, searcher: &mut S) -> Option<()>
    where
        S: TreeSearcher<N>,
    {
        if self.data.dim().get() == searcher.point().len() {
            self.node.iterate_leafs_by(self.data, searcher)
        } else {
            None
        }
    }
    #[inline]
    pub fn find_leaf_of_unit_mut(&mut self, unit: usize) -> Option<&mut Leaf> {
        let v: Box<[N]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, it should always be Some
        self.find_leaf_mut(&v)
    }
    #[inline]
    pub fn find_leaf_mut(&mut self, unit: &[N]) -> Option<&mut Leaf> {
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

impl<N> Node<N> {
    fn new<B, T, S>(config: &B, borders: S, units: &mut [usize]) -> Self
    where
        N: Number,
        B: TreeConfig<N, Data = T, Split = S>,
        T: PointSet<N>,
        S: FindSplit<N, T>,
    {
        // If not enough units remain, a leaf should be constructed
        if units.len() <= config.bucket_size().get() {
            return Leaf::new(units).into();
        }

        // Try to find a split, and if not possible, construct a leaf
        let Some((split, left, right)) = borders.split(config.data(), units) else {
            return Leaf::new(units).into();
        };

        let unit = split.unit();

        Branch {
            split: split.into(),
            left: Self::new(config, left, &mut units[..unit]).into(),
            right: Self::new(config, right, &mut units[unit..]).into(),
        }
        .into()
    }
    fn find_leaf(&self, unit: &[N]) -> &Leaf
    where
        N: Number,
    {
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
    fn find_leaf_mut(&mut self, unit: &[N]) -> &mut Leaf
    where
        N: Number,
    {
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
        N: Number,
        T: PointSet<N>,
        S: TreeSearcher<N>,
    {
        match self {
            Self::Branch(branch) => {
                let (is_left, abs_distance) = branch.split.unit_abs_distance(searcher.point());
                let other = if is_left {
                    branch.left.iterate_leafs_by(data, searcher)?;
                    &branch.right
                } else {
                    branch.right.iterate_leafs_by(data, searcher)?;
                    &branch.left
                };

                if !searcher.is_satisfied(abs_distance) {
                    other.iterate_leafs_by(data, searcher)?;
                }
            }
            Self::Leaf(leaf) => {
                if !leaf.units.is_empty() {
                    searcher.visit_leaf(data, &leaf.units);
                }
            }
        }
        Some(())
    }
}
impl<N> Branch<N> {
    #[inline]
    pub fn split(&self) -> &Split<N> { &self.split }
    #[inline]
    pub fn left(&self) -> &Node<N> { &self.left }
    #[inline]
    pub fn right(&self) -> &Node<N> { &self.right }
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

impl<N> From<Branch<N>> for Node<N>
where
    N: num_traits::Num + Copy,
{
    fn from(branch: Branch<N>) -> Self { Node::Branch(branch) }
}
impl<N> From<Leaf> for Node<N>
where
    N: num_traits::Num + Copy,
{
    fn from(leaf: Leaf) -> Self { Node::Leaf(leaf) }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::kd_tree::{
        PointSet,
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
