use super::node::Node;
use super::split_methods::FindSplit;
use super::store::Store;
use crate::matrix::Matrix;

pub struct Tree<'a> {
    root: Node,
    store: Store<'a>,
    data: &'a Matrix<'a>,
}

impl<'a> Tree<'a> {
    #[inline]
    pub fn new(data: &'a Matrix, find_split: FindSplit, bucket_size: usize) -> Self {
        Tree {
            root: Node::new(find_split, bucket_size, data),
            store: Store::new(data),
            data: data,
        }
    }

    #[inline]
    pub fn find_neighbours_of_id(&mut self, id: usize, n_neighbours: usize) -> &mut Self {
        self.store.set_unit_from_id(id);
        self.root.find_neighbours(&mut self.store, n_neighbours);
        self
    }

    #[inline]
    pub fn get_neighbours(&self) -> &[usize] {
        assert!(self.store.len() > 0, "no neighbours found");
        self.store.get_neighbours()
    }

    #[inline]
    pub fn population_size(&self) -> usize {
        self.data.nrow()
    }

    #[inline]
    pub fn remove_unit(&mut self, id: usize) -> &mut Self {
        assert!(id < self.data.nrow(), "invalid id {}", id);
        self.root.remove_unit(id, self.data);
        self
    }
}
