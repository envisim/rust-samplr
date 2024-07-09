use super::searcher::TreeSearch;
use super::split_methods::FindSplit;
use crate::generate_random::GenerateRandom;
use crate::matrix::OperateMatrix;
use crate::sampling_controller::AccessDataController;

struct NodeSplit {
    dimension: usize,
    value: f64,
    left_child: Box<Node>,
    right_child: Box<Node>,
}

pub struct Node {
    // Non-leaf
    split: Option<NodeSplit>,

    // Leaf
    units: Option<Vec<usize>>,

    // Common
    min_border: Vec<f64>,
    max_border: Vec<f64>,
}

impl Node {
    pub fn new(
        find_split: FindSplit,
        bucket_size: usize,
        data: &dyn OperateMatrix,
        units: &mut [usize],
    ) -> Self {
        assert!(bucket_size > 0);

        let mut node = Node {
            // Non-leaf
            split: None,

            // Leaf
            units: None,

            // Common
            min_border: Vec::<f64>::with_capacity(data.ncol()),
            max_border: Vec::<f64>::with_capacity(data.ncol()),
        };

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

            node.min_border.push(min);
            node.max_border.push(max);
        }

        if node_size_is_zero || units.len() <= bucket_size {
            node.set_units(units);
            return node;
        }

        let (split_unit, split_dimension, split_value) =
            find_split(&node.min_border, &node.max_border, data, &mut *units);

        if split_unit == 0 || split_unit >= units.len() {
            node.set_units(units);
            return node;
        }

        assert!(split_dimension < data.ncol());

        node.split = Some(NodeSplit {
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
        });

        node
    }

    pub fn new_from_controller<R: GenerateRandom, C: AccessDataController<R>>(
        find_split: FindSplit,
        bucket_size: usize,
        controller: &C,
    ) -> Self {
        let mut units: Vec<usize> = controller.indices().get_list().to_vec();
        Node::new(find_split, bucket_size, controller.data(), &mut *units)
    }

    fn set_units(&mut self, units: &[usize]) -> &Self {
        let mut node_units = Vec::<usize>::with_capacity(units.len());
        node_units.extend_from_slice(units);
        self.units = Some(node_units);
        self
    }

    pub fn remove_unit<R: GenerateRandom, C: AccessDataController<R>>(
        &mut self,
        id: usize,
        controller: &C,
    ) {
        if let Some(split) = self.split.as_mut() {
            unsafe {
                // ID is checked by tree
                let distance =
                    *controller.data().get_unchecked((id, split.dimension)) - split.value;

                if distance <= 0.0 {
                    split.left_child.remove_unit(id, controller);
                } else {
                    split.right_child.remove_unit(id, controller);
                }

                return;
            }
        }

        let units = self
            .units
            .as_mut()
            .expect("split is None, thus units must be Some");
        let units_index_opt = units.iter().position(|&p| p == id);

        if let Some(units_index) = units_index_opt {
            units.swap_remove(units_index);
            return;
        }

        panic!("unit {} not found", id);
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

    pub fn find_neighbours<S: TreeSearch, R: GenerateRandom, C: AccessDataController<R>>(
        &self,
        searcher: &mut S,
        controller: &C,
    ) {
        if self.split.is_none() {
            let units: &[usize] = self
                .units
                .as_ref()
                .expect("split is None, thus units must be Some");
            searcher.set_neighbours_from_ids(units, controller);
            return;
        }

        let split = self.split.as_ref().unwrap();
        let unit_value = unsafe { *searcher.unit().get_unchecked(split.dimension) };
        let dist = unit_value - split.value;

        let (first_node, second_node) = if dist <= 0.0 {
            (&split.left_child, &split.right_child)
        } else {
            (&split.right_child, &split.left_child)
        };

        if !searcher.is_satisfied()
            || first_node
                .distance_to_box_in_dimension(split.dimension, unit_value)
                .powi(2)
                <= searcher.max_distance()
        {
            first_node.find_neighbours(searcher, controller);
        }

        if !searcher.is_satisfied()
            || second_node
                .distance_to_box_in_dimension(split.dimension, unit_value)
                .powi(2)
                <= searcher.max_distance()
        {
            second_node.find_neighbours(searcher, controller);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::searcher::SearcherForNeighbours;
    use super::*;
    use crate::{
        generate_random::StaticRandom, kd_tree::split_methods::midpoint_slide, matrix::RefMatrix,
        sampling_controller::*,
    };

    const RAND00: StaticRandom = StaticRandom::new(0.0);
    // const RAND99: StaticRandom = StaticRandom::new(0.999);

    const DATA_10_2: [f64; 20] = [
        0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
        0.66079779, 0.62911404, 0.06178627, //
        0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
        0.3800352, 0.7774452,
    ];

    fn data_10_2<'a>() -> (RefMatrix<'a>, [f64; 10]) {
        (RefMatrix::new(&DATA_10_2, 10), [0.2f64; 10])
    }

    const DATA_9_1: [f64; 9] = [2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2];

    fn data_9_1<'a>() -> (RefMatrix<'a>, [f64; 9]) {
        (RefMatrix::new(&DATA_9_1, 9), [0.1f64; 9])
    }

    #[test]
    fn find() {
        let (data, prob) = data_10_2();
        let controller = DataController::new(&RAND00, &prob, 1e-12, &data);
        let mut searcher = SearcherForNeighbours::new(data.dim(), 1);
        let root = Node::new_from_controller(midpoint_slide, 2, &controller);

        let closest_neighbour: [usize; 10] = [1, 0, 8, 5, 9, 3, 5, 2, 3, 4];
        let mut result: [usize; 10] = [0; 10];

        for i in 0usize..10 {
            searcher.reset_neighbours();
            searcher.set_unit_from_id(i, &controller);
            root.find_neighbours(&mut searcher, &controller);
            result[i] = searcher.neighbours()[0];
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn find_and_remove() {
        let (data, prob) = data_10_2();
        let controller = DataController::new(&RAND00, &prob, 1e-12, &data);
        let mut searcher = SearcherForNeighbours::new(data.dim(), 1);
        let mut root = Node::new_from_controller(midpoint_slide, 2, &controller);

        let closest_neighbour: [usize; 9] = [1, 8, 8, 5, 9, 6, 7, 8, 9];
        let mut result: [usize; 9] = [0; 9];

        for i in 0usize..9 {
            searcher.reset_neighbours();
            searcher.set_unit_from_id(i, &controller);
            root.find_neighbours(&mut searcher, &controller);
            result[i] = searcher.neighbours()[0];
            root.remove_unit(i, &controller);
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn node_splits() {
        let (data, prob) = data_9_1();
        let controller = DataController::new(&RAND00, &prob, 1e-12, &data);

        let mut n0 = Node::new_from_controller(midpoint_slide, 3, &controller);

        let n0s = n0.split.as_mut().expect("");

        assert_eq!(n0s.dimension, 0);
        assert!(n0s.value >= 2.2 && n0s.value <= 10.0);

        let n00 = &mut n0s.left_child;
        let n01 = &mut n0s.right_child;

        let n00s = n00.split.as_mut().expect("");

        assert_eq!(n00s.dimension, 0);
        assert!(n00s.value >= 1.2 && n00s.value <= 2.0);

        let n000 = &mut n00s.left_child;
        let n001 = &mut n00s.right_child;

        let n01u = n01.units.as_mut().expect("");
        let n000u = n000.units.as_mut().expect("");
        let n001u = n001.units.as_mut().expect("");
        n01u.sort();
        n000u.sort();
        n001u.sort();

        assert_eq!(*n000u, vec![6usize, 7, 8]);
        assert_eq!(*n001u, vec![0usize, 1, 2]);
        assert_eq!(*n01u, vec![3usize, 4, 5]);
    }
}
