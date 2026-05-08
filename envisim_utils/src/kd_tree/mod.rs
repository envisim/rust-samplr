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

/// Tree construction trait
///
/// Provides methods necessary for the construction of a [`Tree`]
pub trait TreeConfig<N>
where
    N: Number,
{
    type Data: PointSet<N>;
    type Split: FindSplit<N, Self::Data>;
    #[must_use]
    fn data(&self) -> &Self::Data;
    #[must_use]
    fn bucket_size(&self) -> NonZeroUsize;
    #[must_use]
    fn split_method(&self, units: &[usize]) -> Self::Split;
}

/// A kd-tree
#[must_use]
#[derive(Clone, Debug)]
pub struct Tree<'bdata, N, T> {
    /// The first node
    node: Node<N>,
    /// A reference to the data
    data: &'bdata T,
}

/// A `Node` is either a [`Branch`], which defines a split, or a [`Leaf`] with units.
#[must_use]
#[derive(Clone, Debug)]
#[expect(
    clippy::exhaustive_enums,
    reason = "any new node-types is a breaking change"
)]
pub enum Node<N> {
    Branch(Branch<N>),
    Leaf(Leaf),
}

/// A `Branch` is a [`Node`] which defines a split along a dimension.
#[must_use]
#[derive(Clone, Debug)]
pub struct Branch<N> {
    /// The split of the branch
    split: Split<N>,
    /// The node to the left of the split
    left: Box<Node<N>>,
    /// The node to the right of the split
    right: Box<Node<N>>,
}

/// A `Leaf` is a [`Node`] which contains units.
#[must_use]
#[derive(Clone, Debug)]
pub struct Leaf {
    /// The units contained within the leaf
    units: Vec<usize>,
}

impl<'bdata, N, T> Tree<'bdata, N, T>
where
    N: Number,
    T: PointSet<N>,
{
    /// Constructs a new tree containing `units`, according to some `config`.
    #[inline]
    pub fn new<C, S>(config: &'bdata C, units: &mut [usize]) -> Self
    where
        C: TreeConfig<N, Data = T, Split = S>,
        S: FindSplit<N, T>,
    {
        let data = config.data();
        let borders = config.split_method(units);
        let node = Node::new(config, borders, units);

        Self { node, data }
    }
    /// Returns a reference to the data.
    #[must_use]
    #[inline]
    pub fn data(&self) -> &T { self.data }
    /// Returns a reference to the leaf that would contain `unit`.
    /// Returns `None` if `unit` does not exists in the tree data.
    #[must_use]
    #[inline]
    pub fn find_leaf_of_unit(&self, unit: usize) -> Option<&Leaf> {
        let v: Box<[N]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, find_leaf should always be Some as there can't be
        // dimension mismatch
        self.find_leaf(&v)
    }
    /// Returns a reference to the leaf that would contain `unit`.
    /// Returns `None` if the dimension of `unit` does not match the dimension of the tree data.
    #[must_use]
    #[inline]
    pub fn find_leaf(&self, unit: &[N]) -> Option<&Leaf> {
        (unit.len() == self.data.dim().get()).then(|| self.node.find_leaf(unit))
    }
    /// Iterates the leaf by a [`TreeSearcher`].
    #[must_use]
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
    /// Returns a mutable reference to the leaf that would contain `unit`.
    /// Returns `None` if `unit` does not exists in the tree data.
    #[must_use]
    #[inline]
    pub fn find_leaf_of_unit_mut(&mut self, unit: usize) -> Option<&mut Leaf> {
        let v: Box<[N]> = self.data.to_boxed_slice(unit)?;
        // Since data constructs the slice, it should always be Some
        self.find_leaf_mut(&v)
    }
    /// Returns a mutable reference to the leaf that would contain `unit`.
    /// Returns `None` if the dimension of `unit` does not match the dimension of the tree data.
    #[must_use]
    #[inline]
    pub fn find_leaf_mut(&mut self, unit: &[N]) -> Option<&mut Leaf> {
        (unit.len() == self.data.dim().get()).then(|| self.node.find_leaf_mut(unit))
    }
    /// Inserts a unit into the tree.
    /// Returns `None` if `unit` does not exists in the tree data.
    /// Returns `true` if the unit did not already exist.
    /// Does not rebalance the tree.
    #[inline]
    pub fn insert_unit(&mut self, unit: usize) -> Option<bool> {
        self.find_leaf_of_unit_mut(unit)?.insert_unit(unit).into()
    }
    /// Removes a unit from the tree.
    /// Returns `None` if `unit` does not exists in the tree data.
    /// Returns `false` if the unit did not already exist.
    /// Does not rebalance the tree.
    #[inline]
    pub fn remove_unit(&mut self, unit: usize) -> Option<bool> {
        self.find_leaf_of_unit_mut(unit)?.remove_unit(unit).into()
    }
}

impl<N> Node<N> {
    /// Constructs a new node, by trying to find a possible split, otherwise creating a leaf.
    /// A leaf is also created if the bucket size has been fulfilled.
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

        let unit = split.unit;

        Branch {
            split: split.into(),
            left: Self::new(config, left, &mut units[..unit]).into(),
            right: Self::new(config, right, &mut units[unit..]).into(),
        }
        .into()
    }
    /// Returns a reference to the leaf that would contain `unit`.
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
    /// Returns a mutable reference to the leaf that would contain `unit`.
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
    /// Iterates the leaf by a [`TreeSearcher`].
    #[must_use]
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
    /// Returns the split that defines the branch.
    #[inline]
    pub fn split(&self) -> &Split<N> { &self.split }
    /// Returns a reference to the node left of the split.
    #[inline]
    pub fn left(&self) -> &Node<N> { &self.left }
    /// Returns a reference to the node right of the split.
    #[inline]
    pub fn right(&self) -> &Node<N> { &self.right }
}
impl Leaf {
    /// Constructs a new leaf.
    #[inline]
    fn new(units: &[usize]) -> Self {
        Self {
            units: units.to_vec(),
        }
    }
    /// Returns a reference to the units in the leaf.
    #[must_use]
    #[inline]
    pub fn units(&self) -> &[usize] { &self.units }
    /// Returns `true` if the leaf contains `unit`.
    #[must_use]
    #[inline]
    pub fn contains_unit(&self, unit: usize) -> bool { self.units.contains(&unit) }
    /// Inserts a unit into the leaf.
    /// Returns `true` if the unit did not already exist.
    #[inline]
    fn insert_unit(&mut self, unit: usize) -> bool {
        if !self.contains_unit(unit) {
            self.units.push(unit);
            return true;
        }
        false
    }
    /// Removes a unit from the leaf.
    /// Returns `false` if the unit did not already exist.
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
    #[inline]
    fn from(branch: Branch<N>) -> Self { Node::Branch(branch) }
}
impl<N> From<Leaf> for Node<N>
where
    N: num_traits::Num + Copy,
{
    #[inline]
    fn from(leaf: Leaf) -> Self { Node::Leaf(leaf) }
}

#[cfg(test)]
mod tests {
    use searcher::NearestNeighbourSearcher;

    use super::*;
    use crate::matrix::Matrix;
    use crate::sampling_options::SpreadingOptions;
    use crate::test_utils::*;

    /// Setup a 2D Matrix with 4 points: (0,0), (1,0), (0,1), (1,1)
    fn setup() -> SpreadingOptions<Matrix<f64>> {
        let data = vec![
            0.0, 1.0, 0.0, 1.0, // Dim 0 (X)
            0.0, 0.0, 1.0, 1.0, // Dim 1 (Y)
        ];
        let mat = Matrix::new(data, nz(4)).unwrap();
        // SpreadingOptions implements TreeConfig and uses MidpointSlide internally
        let options = SpreadingOptions::new(mat).unwrap();
        options
    }

    #[test]
    fn test_tree_find_leaf_logic() {
        let options = setup();
        let mut units = vec![0, 1, 2, 3];

        // Construct tree with small bucket size to force branching
        let tree = Tree::new(&options, &mut units);

        // Find leaf by specific coordinate (0.1, 0.1)
        let leaf = tree.find_leaf(&[0.1, 0.1]).expect("Leaf should exist");

        // Given (0,0) is unit 0, and midpoint of (0,1) is 0.5,
        // unit 0 should be in this leaf.
        assert!(leaf.contains_unit(0));

        // Find leaf of a specific unit ID
        let leaf_of_unit = tree.find_leaf_of_unit(3).expect("Unit 3 exists");
        assert!(leaf_of_unit.contains_unit(3));
    }

    #[test]
    fn test_tree_dynamic_modification() {
        let options = setup();
        let mut units = vec![0, 1]; // Start with only 2 units
        let mut tree = Tree::new(&options, &mut units);

        // Insert unit 2 (0, 1)
        let inserted = tree.insert_unit(2).expect("Unit 2 is in Matrix bounds");
        assert!(inserted);

        // Verify it was actually added to the leaf responsible for that area
        let leaf = tree.find_leaf_of_unit(2).unwrap();
        assert!(leaf.contains_unit(2));

        // Remove unit 0
        let removed = tree.remove_unit(0).expect("Unit 0 is in Matrix bounds");
        assert!(removed);
        assert!(!tree.find_leaf_of_unit(0).unwrap().contains_unit(0));
    }

    #[test]
    fn test_tree_iteration_with_real_searcher() {
        let options = setup();
        let mut units = vec![0, 1, 2, 3];
        let tree = Tree::new(&options, &mut units);

        // Use the real NearestNeighbourSearcher
        let mut searcher = NearestNeighbourSearcher::from_slice(&[0.1, 0.1]).unwrap();

        // Traverse the tree
        tree.iterate_leafs_by(&mut searcher)
            .expect("Dimensions match");

        let neighbours = searcher.neighbours();
        assert!(!neighbours.is_empty());
        // Closest to (0.1, 0.1) should be unit 0 (0,0)
        assert_eq!(neighbours[0].id(), 0);
    }

    #[test]
    fn test_tree_dimension_mismatch_safety() {
        let options = setup();
        let mut units = vec![0, 1];
        let tree = Tree::new(&options, &mut units);

        // Try to find leaf using a 3D point on a 2D tree
        let result = tree.find_leaf(&[0.0, 0.0, 0.0]);
        assert!(result.is_none()); // Should safely return None
    }
}
