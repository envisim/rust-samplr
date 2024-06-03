use crate::matrix::Matrix;
// use std::ops::Range;

pub struct Store<'a> {
    id: usize,
    unit: Vec<f64>,
    neighbours: Vec<usize>,
    distances: Vec<f64>,
    data: &'a Matrix<'a>,
}

impl<'a> Store<'a> {
    #[inline]
    pub fn new(data: &'a Matrix) -> Self {
        let (capacity, p) = data.dim();
        let unit: Vec<f64> = vec![0.0f64; p];
        let distances: Vec<f64> = vec![0.0f64; capacity];

        Store {
            id: capacity,
            unit: unit,
            neighbours: Vec::<usize>::with_capacity(capacity),
            distances: distances,
            data: data,
        }
    }

    // #[inline]
    // pub fn add(&mut self, id: usize) -> &mut Self {
    //     self.neighbours.push(id);
    //     self
    // }

    #[inline]
    pub fn add_and_reset(&mut self, id: usize) -> &mut Self {
        self.neighbours.clear();
        self.neighbours.push(id);
        self
    }

    #[inline]
    pub fn add_with_distance(&mut self, id: usize, distance: f64) -> &mut Self {
        self.neighbours.push(id);
        self.set_distance(id, distance)
    }

    #[inline]
    pub fn get_distance_at(&self, k: usize) -> f64 {
        let id = self.neighbours[k];
        self.distances[id]
    }

    // #[inline]
    // pub fn get_distance(&self, id: usize) -> f64 {
    //     self.distances[id]
    // }

    // #[inline]
    // pub fn get_id(&self) -> usize {
    //     self.id
    // }

    #[inline]
    pub fn get_neighbours(&self) -> &[usize] {
        &self.neighbours
    }

    // #[inline]
    // pub fn get_unit(&self) -> &[f64] {
    //     &self.unit
    // }

    #[inline]
    pub fn get_unit_k(&self, k: usize) -> f64 {
        unsafe {
            return *self.unit.get_unchecked(k);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.neighbours.len()
    }

    #[inline]
    pub fn max_distance(&self) -> f64 {
        match self.neighbours.last() {
            Some(i) => self.distances[*i],
            None => f64::INFINITY,
        }
    }

    // #[inline]
    // pub fn min_distance(&self) -> f64 {
    //     match self.neighbours.first() {
    //         Some(i) => self.distances[*i],
    //         None => f64::INFINITY,
    //     }
    // }

    // #[inline]
    // pub fn reserve(&mut self, capacity: usize) -> &mut Self {
    //     let len: usize = self.neighbours.len();

    //     let reserve_additional: usize = if len >= capacity { 0 } else { capacity - len };
    //     self.neighbours.reserve_exact(reserve_additional);
    //     self.distances.resize(capacity, 0.0);
    //     self
    // }

    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.neighbours.clear();
        self
    }

    #[inline]
    pub fn set_unit_from_id(&mut self, id: usize) -> &mut Self {
        self.reset();
        self.id = id;
        self.data
            .into_unit_iter(id)
            .enumerate()
            .for_each(|(i, v)| self.unit[i] = v);
        self
    }

    // #[inline]
    // pub fn set_unit_from_vec(&mut self, unit: &[f64]) -> &mut Self {
    //     assert!(self.unit.len() == unit.len());
    //     self.reset();
    //     self.id = self.data.nrow(); // More than enough
    //     unit.iter().enumerate().for_each(|(i, &v)| self.unit[i] = v);
    //     self
    // }

    #[inline]
    pub fn set_distance(&mut self, id: usize, distance: f64) -> &mut Self {
        self.distances[id] = distance;
        self
    }

    // #[inline]
    // pub fn sort(&mut self, range: Range<usize>) -> &mut Self {
    //     self.neighbours[range]
    //         .sort_unstable_by(|a, b| self.distances[*a].partial_cmp(&self.distances[*b]).unwrap());
    //     self
    // }

    #[inline]
    fn sort_all(&mut self) -> &mut Self {
        self.neighbours
            .sort_unstable_by(|a, b| self.distances[*a].partial_cmp(&self.distances[*b]).unwrap());
        self
    }

    #[inline]
    fn truncate_to_size(&mut self, size: usize) -> &mut Self {
        let mut i: usize = 1;
        let len: usize = self.neighbours.len();

        while i < len {
            if i >= size && self.get_distance_at(i - 1) < self.get_distance_at(i) {
                break;
            }

            i += 1;
        }

        self.neighbours.truncate(i);
        self
    }

    pub fn set_neighbours_from_ids(&mut self, ids: &[usize], n_neighbours: usize) -> &mut Self {
        if ids.len() == 0 {
            return self;
        }

        if n_neighbours == 1 {
            unsafe {
                self.set_neighbours_from_ids_1(ids);
            }
            return self;
        }

        let original_length = self.len();

        if original_length < n_neighbours {
            unsafe {
                self.set_neighbours_from_ids_partial(ids, n_neighbours);
            }
        } else {
            unsafe {
                self.set_neighbours_from_ids_full(ids);
            }
        }

        // If store size hasn't changed, we haven't added any new units
        if self.len() == original_length {
            return self;
        }

        self.sort_all().truncate_to_size(n_neighbours)
    }

    unsafe fn set_neighbours_from_ids_1(&mut self, ids: &[usize]) {
        let store_id = self.id;
        let mut current_max = self.max_distance();

        ids.iter().for_each(|&id| {
            if store_id == id {
                return;
            }

            let distance = self.data.get_distance(id, &self.unit);

            if distance < current_max {
                self.add_and_reset(id).set_distance(id, distance);
                current_max = distance;
            } else if distance == current_max {
                self.add_with_distance(id, distance);
            }
        });
    }

    unsafe fn set_neighbours_from_ids_partial(&mut self, ids: &[usize], n_neighbours: usize) {
        let store_id = self.id;
        // The case of when the store isn't filled yet
        // node_max will store the max _added_ distance from the node
        let mut node_max: f64 = 0.0;

        ids.iter().for_each(|&id| {
            if store_id == id {
                return;
            }

            let distance = self.data.get_distance(id, &self.unit);

            // We should add a unit only in two circumstances:
            // - if the store still has room
            // - if the store has no room, but a larger dist has already been added
            if self.len() < n_neighbours {
                self.add_with_distance(id, distance);

                if distance > node_max {
                    node_max = distance;
                }
            } else if distance < node_max {
                self.add_with_distance(id, distance);
            }

            self.add_with_distance(id, distance);
        });
    }

    unsafe fn set_neighbours_from_ids_full(&mut self, ids: &[usize]) {
        let store_id = self.id;
        let current_max: f64 = self.max_distance();

        ids.iter().for_each(|&id| {
            if store_id == id {
                return;
            }

            let distance = self.data.get_distance(id, &self.unit);

            if distance <= current_max {
                self.add_with_distance(id, distance);
            }
        });
    }
}
