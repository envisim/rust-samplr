use super::{
    node::Node,
    searcher::{SearcherForNeighbours, TreeSearch},
    split_methods::FindSplit,
};
use crate::{
    generate_random::GenerateRandom,
    matrix::{MatrixIterator, OperateMatrix},
    sampling_controller::*,
};

pub struct Tree<'a, R: GenerateRandom, S: TreeSearch> {
    root: Box<Node>,
    searcher: S,
    controller: DataController<'a, R>,
}

impl<'a, R: GenerateRandom, S: TreeSearch> Tree<'a, R, S> {
    pub fn new(
        controller: DataController<'a, R>,
        searcher: S,
        find_split: FindSplit,
        bucket_size: usize,
    ) -> Self {
        Tree {
            root: Box::new(Node::new_from_controller(
                find_split,
                bucket_size,
                &controller,
            )),
            searcher: searcher,
            controller: controller,
        }
    }

    pub fn new_with_root(controller: DataController<'a, R>, searcher: S, root: Box<Node>) -> Self {
        Tree {
            root,
            searcher,
            controller,
        }
    }

    #[inline]
    pub fn find_neighbours_of_id(&mut self, id: usize) {
        self.searcher.set_unit_from_id(id, &self.controller);

        if self.controller.indices().len() == 1 {
            return;
        }

        self.root
            .find_neighbours(&mut self.searcher, &self.controller);
    }

    #[inline]
    pub fn controller(&self) -> &DataController<'a, R> {
        &self.controller
    }

    #[inline]
    pub fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        &mut self.controller
    }

    #[inline]
    pub fn searcher(&self) -> &S {
        &self.searcher
    }

    #[inline]
    pub fn searcher_mut(&mut self) -> &mut S {
        &mut self.searcher
    }

    #[inline]
    pub fn decide_unit(&mut self, id: usize) -> Option<bool> {
        assert!(id < self.controller.data_nrow(), "invalid id {}", id);
        let removed = self.controller.decide_unit(id);

        if removed.is_some() {
            self.root.remove_unit(id, &self.controller);
        }

        removed
    }
}

impl<'a, R: GenerateRandom> Tree<'a, R, SearcherForNeighbours> {
    pub fn find_neighbours_of_unit<M: OperateMatrix>(&mut self, unit: &mut MatrixIterator<'a, M>) {
        self.searcher.set_unit_from_iter(unit);

        self.root
            .find_neighbours(&mut self.searcher, &self.controller);
    }
}
