use envisim_samplr_utils::{
    generate_random::GenerateRandom,
    kd_tree::{midpoint_slide, SearcherForNeighbours, Tree, TreeSearch},
    matrix::OperateMatrix,
    matrix::RefMatrix,
    sampling_controller::*,
};

pub trait RunPivotalMethod<R: GenerateRandom, C: AccessBaseController<R>> {
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    fn draw_units(&mut self) -> Option<(usize, usize)>;
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.controller_mut().decide_unit(id)
    }
    #[inline]
    fn update_probabilities(&mut self, id1: usize, id2: usize) {
        let mut p1 = *self.controller().probability(id1);
        let mut p2 = *self.controller().probability(id2);
        let psum = p1 + p2;

        if psum > 1.0 {
            if 1.0 - p2 > self.controller().random().random_float_scale(2.0 - psum) {
                p1 = 1.0;
                p2 = psum - 1.0;
            } else {
                p1 = psum - 1.0;
                p2 = 1.0;
            }
        } else {
            if p2 > self.controller().random().random_float_scale(psum) {
                p1 = 0.0;
                p2 = psum;
            } else {
                p1 = psum;
                p2 = 0.0;
            }
        }

        *self.controller_mut().probability_mut(id1) = p1;
        *self.controller_mut().probability_mut(id2) = p2;
        self.decide_unit(id1);
        self.decide_unit(id2);
    }

    fn run(&mut self) {
        while let Some((id1, id2)) = self.draw_units() {
            self.update_probabilities(id1, id2);
        }

        if let Some(id) = self.controller_mut().update_last_unit() {
            self.decide_unit(id);
        }
    }
}

pub struct LocalPivotalMethod1<'a, R: GenerateRandom> {
    tree: Tree<'a, R, SearcherForNeighbours>,
    candidates: Vec<usize>,
}

impl<'a, R: GenerateRandom> LocalPivotalMethod1<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        Self {
            tree: Tree::new(
                DataController::new(rand, probabilities, eps, data),
                SearcherForNeighbours::new(data.dim(), 1),
                midpoint_slide,
                bucket_size,
            ),
            candidates: Vec::<usize>::with_capacity(20),
        }
    }

    /// Draw a sample using the local pivotal method, variant 1 (LPM1).
    ///
    /// Grafström, A., Lundström, N.L.P. & Schelin, L. (2012).
    /// Spatially balanced sampling through the Pivotal method.
    /// Biometrics 68(2), 514-520.
    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut lpm = Self::new(rand, probabilities, eps, data, bucket_size);
        lpm.run();
        lpm.controller_mut().sample_sort();
        lpm.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunPivotalMethod<R, DataController<'a, R>>
    for LocalPivotalMethod1<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.tree.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.tree.controller_mut()
    }
    fn draw_units(&mut self) -> Option<(usize, usize)> {
        let len = self.controller().indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            let id1 = *self.controller().indices().get(0).unwrap();
            let id2 = *self.controller().indices().get(1).unwrap();
            return Some((id1, id2));
        }

        loop {
            let id1: usize = *self
                .controller()
                .get_random_index()
                .expect("indices should not be empty");
            self.tree.find_neighbours_of_id(id1);
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.tree.searcher().neighbours());

            let mut i: usize = 0;

            while i < self.candidates.len() {
                self.tree.find_neighbours_of_id(self.candidates[i]);

                if self
                    .tree
                    .searcher()
                    .neighbours()
                    .iter()
                    .any(|&id| id == id1)
                {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if self.candidates.len() > 0 {
                let id2: usize = *self
                    .controller()
                    .random_get(&self.candidates)
                    .expect("candidates should not be empty");
                return Some((id1, id2));
            }
        }
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.tree.decide_unit(id)
    }
}

pub struct LocalPivotalMethod1S<'a, R: GenerateRandom> {
    tree: Tree<'a, R, SearcherForNeighbours>,
    candidates: Vec<usize>,
    history: Vec<usize>,
}

impl<'a, R: GenerateRandom> LocalPivotalMethod1S<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        Self {
            tree: Tree::new(
                DataController::new(rand, probabilities, eps, data),
                SearcherForNeighbours::new(data.dim(), 1),
                midpoint_slide,
                bucket_size,
            ),
            candidates: Vec::<usize>::with_capacity(20),
            history: Vec::<usize>::with_capacity(probabilities.len()),
        }
    }

    /// Draw a sample using the local pivotal method, variant S (LPM1S).
    ///
    /// Prentius, W. (2024)
    /// Manuscript.
    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut lpm = Self::new(rand, probabilities, eps, data, bucket_size);
        lpm.run();
        lpm.controller_mut().sample_sort();
        lpm.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunPivotalMethod<R, DataController<'a, R>>
    for LocalPivotalMethod1S<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.tree.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.tree.controller_mut()
    }
    fn draw_units(&mut self) -> Option<(usize, usize)> {
        let len = self.controller().indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            let id1 = *self.controller().indices().get(0).unwrap();
            let id2 = *self.controller().indices().get(1).unwrap();
            return Some((id1, id2));
        }

        while let Some(&id) = self.history.last() {
            if self.controller().indices().includes(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.len() == 0 {
            self.history.push(
                *self
                    .controller()
                    .get_random_index()
                    .expect("indices should not be empty"),
            );
        }

        loop {
            let id1 = *self.history.last().expect("history should have units");
            self.tree.find_neighbours_of_id(id1);
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.tree.searcher().neighbours());

            let mut i: usize = 0;
            let mut len: usize = self.candidates.len();

            while i < len {
                self.tree.find_neighbours_of_id(self.candidates[i]);

                if self
                    .tree
                    .searcher()
                    .neighbours()
                    .iter()
                    .any(|&id| id == id1)
                {
                    i += 1;
                } else {
                    // If we does not find any compatible matches, we use the candidates to continue our seach
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2: usize = *self
                    .controller()
                    .random_get(&self.candidates[0..len])
                    .expect("candidates should have possible units");
                return Some((id1, id2));
            }

            if self.history.len() == self.controller().data_nrow() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history.push(
                *self
                    .controller()
                    .random_get(&self.candidates)
                    .expect("candidates should have units"),
            );
        }
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.tree.decide_unit(id)
    }
}

pub struct LocalPivotalMethod2<'a, R: GenerateRandom> {
    tree: Tree<'a, R, SearcherForNeighbours>,
}

impl<'a, R: GenerateRandom> LocalPivotalMethod2<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        Self {
            tree: Tree::new(
                DataController::new(rand, probabilities, eps, data),
                SearcherForNeighbours::new(data.dim(), 1),
                midpoint_slide,
                bucket_size,
            ),
        }
    }

    /// Draw a sample using the local pivotal method, variant 2 (LPM2).
    ///
    /// Grafström, A., Lundström, N.L.P. & Schelin, L. (2012).
    /// Spatially balanced sampling through the Pivotal method.
    /// Biometrics 68(2), 514-520.
    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut lpm = Self::new(rand, probabilities, eps, data, bucket_size);
        lpm.run();
        lpm.controller_mut().sample_sort();
        lpm.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunPivotalMethod<R, DataController<'a, R>>
    for LocalPivotalMethod2<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.tree.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.tree.controller_mut()
    }
    fn draw_units(&mut self) -> Option<(usize, usize)> {
        let len = self.controller().indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            let id1 = *self.controller().indices().get(0).unwrap();
            let id2 = *self.controller().indices().get(1).unwrap();
            return Some((id1, id2));
        }

        let id1 = *self
            .controller()
            .get_random_index()
            .expect("indices should not be empty");
        self.tree.find_neighbours_of_id(id1);
        let id2 = *self
            .controller()
            .random_get(self.tree.searcher().neighbours())
            .expect("neighbours should not be empty");

        Some((id1, id2))
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.tree.decide_unit(id)
    }
}

pub struct RandomPivotalMethod<'a, R: GenerateRandom> {
    controller: BaseController<'a, R>,
}

impl<'a, R: GenerateRandom> RandomPivotalMethod<'a, R> {
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64) -> Self {
        Self {
            controller: BaseController::new(rand, probabilities, eps),
        }
    }

    pub fn draw(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize> {
        let mut rpm = Self::new(rand, probabilities, eps);
        rpm.run();
        rpm.controller_mut().sample_sort();
        rpm.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunPivotalMethod<R, BaseController<'a, R>>
    for RandomPivotalMethod<'a, R>
{
    #[inline]
    fn controller(&self) -> &BaseController<'a, R> {
        &self.controller
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut BaseController<'a, R> {
        &mut self.controller
    }
    fn draw_units(&mut self) -> Option<(usize, usize)> {
        let len = self.controller().indices().len();
        if len <= 1 {
            return None;
        }

        let id1 = *self
            .controller()
            .get_random_index()
            .expect("indices should not be empty");
        let mut id2 = *self
            .controller()
            .indices()
            .get(self.controller().random().random_usize_scale(len - 1))
            .expect("indices should not be empty");

        if id1 == id2 {
            id2 = *self
                .controller()
                .indices()
                .get_last()
                .expect("indices should not be empty");
        }

        Some((id1, id2))
    }
}

pub struct SequentialPivotalMethod<'a, R: GenerateRandom> {
    controller: BaseController<'a, R>,
    pair: [usize; 2],
}

impl<'a, R: GenerateRandom> SequentialPivotalMethod<'a, R> {
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64) -> Self {
        Self {
            controller: BaseController::new(rand, probabilities, eps),
            pair: [0, 1],
        }
    }

    pub fn draw(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize> {
        let mut spm = Self::new(rand, probabilities, eps);

        spm.run();
        spm.controller_mut().sample_sort();
        spm.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunPivotalMethod<R, BaseController<'a, R>>
    for SequentialPivotalMethod<'a, R>
{
    #[inline]
    fn controller(&self) -> &BaseController<'a, R> {
        &self.controller
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut BaseController<'a, R> {
        &mut self.controller
    }
    fn draw_units(&mut self) -> Option<(usize, usize)> {
        if self.controller().indices().len() <= 1 {
            return None;
        }

        if !self.controller().indices().includes(self.pair[0]) {
            self.pair[0] = self.pair[1];

            while !self.controller().indices().includes(self.pair[0]) {
                self.pair[0] += 1;

                if self.pair[0] >= self.controller().probabilities().len() {
                    panic!("spm looped past last unit");
                }
            }

            self.pair[1] = self.pair[0] + 1;
        }

        while !self.controller().indices().includes(self.pair[1]) {
            self.pair[1] += 1;

            if self.pair[1] >= self.controller().probabilities().len() {
                panic!("spm looped past last unit");
            }
        }

        Some((self.pair[0], self.pair[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_samplr_utils::generate_random::StaticRandom;

    const RAND00: StaticRandom = StaticRandom::new(0.0);
    const RAND99: StaticRandom = StaticRandom::new(0.999);

    const DATA_10_2: [f64; 20] = [
        0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
        0.66079779, 0.62911404, 0.06178627, //
        0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185, 0.9919061,
        0.3800352, 0.7774452,
    ];

    fn data_10_2<'a>() -> (RefMatrix<'a>, [f64; 10]) {
        (RefMatrix::new(&DATA_10_2, 10), [0.2f64; 10])
    }

    #[test]
    fn lpm1_draw() {
        let (data, prob) = data_10_2();

        let mut lpm00 = LocalPivotalMethod1::new(&RAND00, &prob, 1e-12, &data, 2);
        assert_eq!(lpm00.draw_units(), Some((0, 1)));

        let mut lpm99 = LocalPivotalMethod1::new(&RAND99, &prob, 1e-12, &data, 2);
        assert_eq!(lpm99.draw_units(), Some((9, 4)));
    }

    #[test]
    fn lpm1s() {
        let (data, prob) = data_10_2();

        let mut lpm00 = LocalPivotalMethod1S::new(&RAND99, &prob, 1e-12, &data, 2);
        let mut units: (usize, usize);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (9, 4));
        assert_eq!(lpm00.history, vec![9]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (3, 5));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (3, 8));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (3, 6));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (2, 7));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (2, 3));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (2, 9));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        assert_eq!(units, (1, 0));
        assert_eq!(lpm00.history, vec![9, 2, 8, 3, 2, 1]);
        lpm00.update_probabilities(units.0, units.1);
        units = lpm00.draw_units().unwrap();
        lpm00.update_probabilities(units.0, units.1);

        assert_eq!(lpm00.controller().indices().len(), 0);
        lpm00.controller_mut().sample_sort();
        assert_eq!(lpm00.controller().sample(), vec![1, 3]);
    }

    #[test]
    fn lpm2() {
        let (data, prob) = data_10_2();

        let lpm00_result = LocalPivotalMethod2::draw(&RAND00, &prob, 1e-12, &data, 5);
        assert_eq!(lpm00_result, vec![1, 3]);
    }

    #[test]
    fn rpm() {
        let rpm00_result = RandomPivotalMethod::draw(&RAND00, &[0.5f64; 4], 1e-12);
        assert_eq!(rpm00_result, vec![1, 3]);

        let rpm99_result = RandomPivotalMethod::draw(&RAND99, &[0.5f64; 4], 1e-12);
        assert_eq!(rpm99_result, vec![1, 3]);
    }

    #[test]
    fn spm() {
        let spm00_result = SequentialPivotalMethod::draw(&RAND00, &[0.5f64; 4], 1e-12);
        assert_eq!(spm00_result, vec![1, 3]);

        let spm99_result = SequentialPivotalMethod::draw(&RAND99, &[0.5f64; 4], 1e-12);
        assert_eq!(spm99_result, vec![0, 2]);
    }
}
