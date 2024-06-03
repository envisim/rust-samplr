use super::split_methods::FindSplit;
use super::store::Store;
use crate::matrix::Matrix;

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
    pub fn new_from_units(
        find_split: FindSplit,
        bucket_size: usize,
        data: &Matrix,
        units: &mut [usize],
    ) -> Self {
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
            let mut min = data.get(units[0], k);
            let mut max = min;

            units.iter().for_each(|&id| {
                let val: f64 = data.get(id, k);
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
            left_child: Box::new(Node::new_from_units(
                find_split,
                bucket_size,
                data,
                &mut units[..split_unit],
            )),
            right_child: Box::new(Node::new_from_units(
                find_split,
                bucket_size,
                data,
                &mut units[split_unit..],
            )),
        });

        node
    }

    pub fn new(find_split: FindSplit, bucket_size: usize, data: &Matrix) -> Self {
        let mut units: Vec<usize> = (0usize..data.nrow()).collect();
        Node::new_from_units(find_split, bucket_size, data, &mut *units)
    }

    #[inline]
    fn distance_to_box(&self, split_dimension: usize, value: f64) -> f64 {
        unsafe {
            let min = *self.min_border.get_unchecked(split_dimension);
            let max = *self.max_border.get_unchecked(split_dimension);

            if value <= min {
                return min - value;
            } else if value <= max {
                return 0.0;
            } else {
                return value - max;
            }
        }
    }

    fn set_units(&mut self, units: &[usize]) -> &Self {
        let mut node_units = Vec::<usize>::with_capacity(units.len());
        node_units.extend_from_slice(units);
        self.units = Some(node_units);
        self
    }

    pub fn find_neighbours(&self, store: &mut Store, n_neighbours: usize) -> &Self {
        if self.split.is_none() {
            let members: &[usize] = self
                .units
                .as_ref()
                .expect("split is None, thus units must be Some");

            store.set_neighbours_from_ids(members, n_neighbours);
            return self;
        }

        let split = self.split.as_ref().unwrap();
        let unit_value = store.get_unit_k(split.dimension);
        let dist = unit_value - split.value;

        let (first_node, second_node) = if dist <= 0.0 {
            (&split.left_child, &split.right_child)
        } else {
            (&split.right_child, &split.left_child)
        };

        if store.len() < n_neighbours
            || first_node
                .distance_to_box(split.dimension, unit_value)
                .powi(2)
                <= store.max_distance()
        {
            first_node.find_neighbours(store, n_neighbours);
        }

        if store.len() < n_neighbours
            || second_node
                .distance_to_box(split.dimension, unit_value)
                .powi(2)
                <= store.max_distance()
        {
            second_node.find_neighbours(store, n_neighbours);
        }

        self
    }

    pub fn remove_unit(&mut self, id: usize, data: &Matrix) -> &mut Self {
        if let Some(split) = self.split.as_mut() {
            unsafe {
                // ID is checked by tree
                let distance = data.get_unsafe(id, split.dimension) - split.value;

                if distance <= 0.0 {
                    split.left_child.remove_unit(id, data);
                } else {
                    split.right_child.remove_unit(id, data);
                }

                return self;
            }
        }

        let units = self
            .units
            .as_mut()
            .expect("split is None, thus units must be Some");
        let units_index_opt = units.iter().position(|&p| p == id);

        if let Some(units_index) = units_index_opt {
            units.swap_remove(units_index);
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use crate::kd_tree::split_methods::midpoint_slide;

    use super::*;

    #[test]
    fn find() {
        let data = Matrix::new(
            &[
                0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
                0.66079779, 0.62911404, 0.06178627, //
                0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185,
                0.9919061, 0.3800352, 0.7774452,
            ],
            10,
        );
        let mut store = Store::new(&data);
        let tree = Node::new(midpoint_slide, 2, &data);

        let closest_neighbour: [usize; 10] = [1, 0, 8, 5, 9, 3, 5, 2, 3, 4];
        let mut result: [usize; 10] = [0; 10];

        for i in 0usize..10 {
            store.reset().set_unit_from_id(i);
            tree.find_neighbours(&mut store, 1);
            result[i] = store.get_neighbours()[0];
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn find_and_remove() {
        let data = Matrix::new(
            &[
                0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
                0.66079779, 0.62911404, 0.06178627, //
                0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185,
                0.9919061, 0.3800352, 0.7774452,
            ],
            10,
        );
        let mut store = Store::new(&data);
        let mut tree = Node::new(midpoint_slide, 2, &data);

        let closest_neighbour: [usize; 9] = [1, 8, 8, 5, 9, 6, 7, 8, 9];
        let mut result: [usize; 9] = [0; 9];

        for i in 0usize..9 {
            store.reset().set_unit_from_id(i);
            tree.find_neighbours(&mut store, 1);
            result[i] = store.get_neighbours()[0];
            tree.remove_unit(i, &data);
        }

        assert_eq!(result, closest_neighbour);
    }

    #[test]
    fn node_splits() {
        let data = Matrix::new(&[2.0, 2.1, 2.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2], 9);

        let mut n0 = Node::new(midpoint_slide, 3, &data);

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
