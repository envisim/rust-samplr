use envisim_samplr_utils::{
    generate_random::GenerateRandom,
    kd_tree::{midpoint_slide, SearcherForNeighboursWithWeights, Tree, TreeSearch},
    matrix::{OperateMatrix, RefMatrix},
    sampling_controller::*,
    utils::usize_to_f64,
};

pub trait RunCorrelatedPoisson<R: GenerateRandom, C: AccessBaseController<R>> {
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    fn neighbours(&self) -> &[usize];
    fn weight(&self, id: usize) -> f64;
    fn draw_unit(&mut self) -> Option<usize>;
    fn find_neighbours(&mut self, id: usize);
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.controller_mut().decide_unit(id)
    }
    #[inline]
    fn update_probabilities(&mut self, probability_quota: f64) {
        let mut remaining_weight: f64 = 1.0;
        let mut i: usize = 0;

        while i < self.neighbours().len() && remaining_weight > 0.0 {
            let id = self.neighbours()[i];

            let removable_weight = f64::min(remaining_weight, self.weight(id));

            unsafe {
                self.controller_mut().probabilities_mut()[id] +=
                    removable_weight * probability_quota;
            }
            self.decide_unit(id);
            remaining_weight -= removable_weight;
            i += 1;
        }
    }
    fn run(&mut self) {
        while let Some(id) = self.draw_unit() {
            self.find_neighbours(id);

            let mut initial_probability = *self.controller().probability(id);
            if self.controller().random().random_float_of(id) < initial_probability {
                *self.controller_mut().probability_mut(id) = 1.0;
                initial_probability -= 1.0;
            } else {
                *self.controller_mut().probability_mut(id) = 0.0;
            }

            self.decide_unit(id);
            self.update_probabilities(initial_probability);
        }
    }
}

pub struct SpatiallyCorrelatedPoissonSampling<'a, R: GenerateRandom> {
    tree: Tree<'a, R, SearcherForNeighboursWithWeights>,
}

impl<'a, R: GenerateRandom> SpatiallyCorrelatedPoissonSampling<'a, R> {
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
                SearcherForNeighboursWithWeights::new(data.dim()),
                midpoint_slide,
                bucket_size,
            ),
        }
    }

    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut cps = Self::new(rand, probabilities, eps, data, bucket_size);
        cps.run();
        cps.controller_mut().sample_sort();
        cps.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCorrelatedPoisson<R, DataController<'a, R>>
    for SpatiallyCorrelatedPoissonSampling<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.tree.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.tree.controller_mut()
    }
    #[inline]
    fn neighbours(&self) -> &[usize] {
        self.tree.searcher().neighbours()
    }
    #[inline]
    fn weight(&self, id: usize) -> f64 {
        self.tree.searcher().weight(id)
    }
    fn draw_unit(&mut self) -> Option<usize> {
        self.controller().get_random_index().cloned()
    }
    #[inline]
    fn find_neighbours(&mut self, id: usize) {
        self.tree.find_neighbours_of_id(id);
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.tree.decide_unit(id)
    }
    fn update_probabilities(&mut self, probability_quota: f64) {
        let mut remaining_weight: f64 = 1.0;
        let mut i: usize = 0;

        while i < self.neighbours().len() {
            // Start by adding up the weights of all ties
            let mut sum_of_tie_weights = self.weight(self.neighbours()[i]);
            let distance = self.tree.searcher().distances()[self.neighbours()[i]];
            let mut j: usize = i + 1;

            while j < self.neighbours().len()
                && self.tree.searcher().distances()[self.neighbours()[j]] == distance
            {
                sum_of_tie_weights += self.weight(self.neighbours()[j]);
                j += 1;
            }

            // If the sum of all ties are less than the remaining weight, we can
            // continue as usual
            if sum_of_tie_weights < remaining_weight {
                while i < j {
                    let id = self.tree.searcher().neighbour_k(i);
                    let removable_weight = self.weight(id);
                    unsafe {
                        self.controller_mut().probabilities_mut()[id] +=
                            removable_weight * probability_quota;
                    }
                    self.decide_unit(id);
                    remaining_weight -= removable_weight;
                    i += 1;
                }

                i = j;
                continue;
            }

            // If the sum of all ties are more than the remaining weight, we need
            // to be a bit more tactful.
            // No unit should be able to get more than a "fair" share.
            // Initially, each unit should get equal weight.
            // If some units cannot accept this much weight, the remainder will
            // be redistributed amongst the others.
            // Thus, we sort the remaining neighbours with smallest first.
            let mut sharers = usize_to_f64(j - i);
            self.tree.searcher_mut().sort_neighbours_by_weight(i, j);

            while i < j {
                let id = self.tree.searcher().neighbour_k(i);
                let removable_weight = f64::min(remaining_weight / sharers, self.weight(id));
                unsafe {
                    self.controller_mut().probabilities_mut()[id] +=
                        removable_weight * probability_quota;
                }
                self.decide_unit(id);
                remaining_weight -= removable_weight;
                sharers -= 1.0;
                i += 1;
            }

            i = j;
        }
    }
}

pub struct CoordinatedSpatiallyCorrelatedPoissonSampling<'a, R: GenerateRandom> {
    scps: SpatiallyCorrelatedPoissonSampling<'a, R>,
    unit: usize,
}

impl<'a, R: GenerateRandom> CoordinatedSpatiallyCorrelatedPoissonSampling<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        Self {
            scps: SpatiallyCorrelatedPoissonSampling::new(
                rand,
                probabilities,
                eps,
                data,
                bucket_size,
            ),
            unit: 0,
        }
    }

    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut cps = Self::new(rand, probabilities, eps, data, bucket_size);
        cps.run();
        cps.controller_mut().sample_sort();
        cps.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCorrelatedPoisson<R, DataController<'a, R>>
    for CoordinatedSpatiallyCorrelatedPoissonSampling<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.scps.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.scps.controller_mut()
    }
    #[inline]
    fn neighbours(&self) -> &[usize] {
        self.scps.neighbours()
    }
    #[inline]
    fn weight(&self, id: usize) -> f64 {
        self.scps.weight(id)
    }
    fn draw_unit(&mut self) -> Option<usize> {
        if self.controller().indices().len() == 0 {
            return None;
        }

        while self.unit < self.controller().population_size() {
            if self.controller().indices().includes(self.unit) {
                return Some(self.unit);
            }

            self.unit += 1;
        }

        return None;
    }
    #[inline]
    fn find_neighbours(&mut self, id: usize) {
        self.scps.find_neighbours(id);
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.scps.decide_unit(id)
    }
    #[inline]
    fn update_probabilities(&mut self, probability_quota: f64) {
        self.scps.update_probabilities(probability_quota);
    }
}

pub struct LocallyCorrelatedPoissonSampling<'a, R: GenerateRandom> {
    scps: SpatiallyCorrelatedPoissonSampling<'a, R>,
    candidates: Vec<usize>,
}

impl<'a, R: GenerateRandom> LocallyCorrelatedPoissonSampling<'a, R> {
    pub fn new(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Self {
        Self {
            scps: SpatiallyCorrelatedPoissonSampling::new(
                rand,
                probabilities,
                eps,
                data,
                bucket_size,
            ),
            candidates: Vec::<usize>::with_capacity(probabilities.len()),
        }
    }

    pub fn draw(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize> {
        let mut cps = Self::new(rand, probabilities, eps, data, bucket_size);
        cps.run();
        cps.controller_mut().sample_sort();
        cps.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCorrelatedPoisson<R, DataController<'a, R>>
    for LocallyCorrelatedPoissonSampling<'a, R>
{
    #[inline]
    fn controller(&self) -> &DataController<'a, R> {
        self.scps.controller()
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut DataController<'a, R> {
        self.scps.controller_mut()
    }
    #[inline]
    fn neighbours(&self) -> &[usize] {
        self.scps.neighbours()
    }
    #[inline]
    fn weight(&self, id: usize) -> f64 {
        self.scps.weight(id)
    }
    fn draw_unit(&mut self) -> Option<usize> {
        if self.controller().indices().len() <= 1 {
            return self.controller().indices().get_first().cloned();
        }

        let mut minimum_distance = f64::MAX;

        // Loop through all remaining units
        let mut i = 0;
        while i < self.controller().indices().len() {
            let id: usize = *self
                .controller()
                .indices()
                .get(i)
                .expect("id should still have units");
            // while let Some(id) = self.controller().indices().get(i) {
            self.find_neighbours(id);
            let distance = self.scps.tree.searcher().max_distance();

            if distance < minimum_distance {
                self.candidates.clear();
                self.candidates.push(id);
                minimum_distance = distance;
            } else if distance == minimum_distance {
                self.candidates.push(id);
            }

            i += 1;
        }

        self.controller()
            .random()
            .random_get(&self.candidates)
            .cloned()
    }
    #[inline]
    fn find_neighbours(&mut self, id: usize) {
        self.scps.find_neighbours(id);
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.scps.decide_unit(id)
    }
    #[inline]
    fn update_probabilities(&mut self, probability_quota: f64) {
        self.scps.update_probabilities(probability_quota);
    }
}

pub struct CorrelatedPoissonSampling<'a, R: GenerateRandom> {
    controller: BaseController<'a, R>,
    neighbours: Vec<usize>,
    weights: Vec<f64>,
    unit: usize,
}

impl<'a, R: GenerateRandom> CorrelatedPoissonSampling<'a, R> {
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64) -> Self {
        Self {
            controller: BaseController::new(rand, probabilities, eps),
            neighbours: Vec::<usize>::with_capacity(probabilities.len()),
            weights: vec![0.0f64; probabilities.len()],
            unit: 0,
        }
    }

    pub fn draw(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize> {
        let mut cps = Self::new(rand, probabilities, eps);
        cps.run();
        cps.controller_mut().sample_sort();
        cps.controller().sample().to_vec()
    }
}

impl<'a, R: GenerateRandom> RunCorrelatedPoisson<R, BaseController<'a, R>>
    for CorrelatedPoissonSampling<'a, R>
{
    #[inline]
    fn controller(&self) -> &BaseController<'a, R> {
        &self.controller
    }
    #[inline]
    fn controller_mut(&mut self) -> &mut BaseController<'a, R> {
        &mut self.controller
    }
    #[inline]
    fn neighbours(&self) -> &[usize] {
        &self.neighbours
    }
    fn weight(&self, id: usize) -> f64 {
        self.weights[id]
    }
    fn draw_unit(&mut self) -> Option<usize> {
        if self.controller().indices().len() == 0 {
            return None;
        }

        while self.unit < self.controller().population_size() {
            if self.controller().indices().includes(self.unit) {
                return Some(self.unit);
            }

            self.unit += 1;
        }

        return None;
    }
    fn find_neighbours(&mut self, id: usize) {
        self.neighbours.clear();
        let mut nid = id + 1;
        let mut total_weight = 0.0;

        while nid < self.controller().population_size() && total_weight < 1.0 {
            if self.controller().indices().includes(nid) {
                let weight = self.controller().weight(id, nid);
                self.weights[nid] = weight;
                total_weight += weight;
                self.neighbours.push(nid);
            }

            nid += 1;
        }
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        self.controller.decide_unit(id)
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

    macro_rules! assert_delta {
        ($a:expr,$b:expr,$d:expr) => {
            assert!(($a - $b).abs() < $d);
        };
    }

    #[test]
    fn scps_0() {
        let (data, prob) = data_10_2();

        let mut scps00 = SpatiallyCorrelatedPoissonSampling::new(&RAND00, &prob, 1e-12, &data, 2);
        assert_eq!(scps00.draw_unit(), Some(0));
        assert_delta!(scps00.controller().probability(0), 0.2, 1e-12);
        scps00.find_neighbours(0);
        // 0 1 8 4 2 9 3 5 6 7
        *scps00.controller_mut().probability_mut(0) = 0.0;
        assert_eq!(scps00.decide_unit(0), Some(false));
        assert_eq!(scps00.neighbours().len(), 4);
        assert_eq!(scps00.neighbours(), vec![1, 8, 4, 2]);
        scps00.update_probabilities(0.2);
        assert_delta!(scps00.controller().probability(1), 0.25, 1e-12);
        assert_delta!(scps00.controller().probability(8), 0.25, 1e-12);
        assert_delta!(scps00.controller().probability(4), 0.25, 1e-12);
        assert_delta!(scps00.controller().probability(2), 0.25, 1e-12);

        let s = SpatiallyCorrelatedPoissonSampling::draw(&RAND00, &prob, 1e-12, &data, 2);
        assert_eq!(s, vec![0, 9]);
    }

    #[test]
    fn scps_1() {
        let (data, prob) = data_10_2();

        let mut scps00 = SpatiallyCorrelatedPoissonSampling::new(&RAND99, &prob, 1e-12, &data, 2);
        assert_eq!(scps00.draw_unit(), Some(9));
        assert_delta!(scps00.controller().probability(9), 0.2, 1e-12);
        scps00.find_neighbours(9);
        // 9 4 2 0 7 1 8 5 6 3
        *scps00.controller_mut().probability_mut(9) = 1.0;
        assert_eq!(scps00.decide_unit(9), Some(true));
        assert_eq!(scps00.neighbours().len(), 4);
        assert_eq!(scps00.neighbours(), vec![4, 2, 0, 7]);
        scps00.update_probabilities(-0.8);
        assert_delta!(scps00.controller().probability(4), 0.0, 1e-12);
        assert_delta!(scps00.controller().probability(2), 0.0, 1e-12);
        assert_delta!(scps00.controller().probability(0), 0.0, 1e-12);
        assert_delta!(scps00.controller().probability(7), 0.0, 1e-12);

        let s = SpatiallyCorrelatedPoissonSampling::draw(&RAND99, &prob, 1e-12, &data, 2);
        assert_eq!(s, vec![0, 2]);
        // 0    1    2    3    4    5    6    7  8  9
        // 20   20   20   20   20   20   20   20 20 20
        // 25   20   25   20   25   20   20   25 20  0
        // 25   2375 3125 25   25   25   20   25  0  0
        // 25   2375 4167 25   3292 25   2667 0   0  0
        // 25   2375 5016 3409 3292 3409 0    0   0  0
        // 25   2375 6662 5172 3292 0    0    0   0  0
        // 2523 2375 9931 5172 0    0    0    0   0  0
        // 5081 4919 +    0    0    0    0    0   0  0
        // +    0    +    0    0    0    0    0   0  0
    }

    #[test]
    fn cps() {
        let (_, prob) = data_10_2();

        let mut cps00 = CorrelatedPoissonSampling::new(&RAND00, &prob, 1e-12);
        assert_eq!(cps00.draw_unit(), Some(0));
        assert_delta!(cps00.controller().probability(0), 0.2, 1e-12);
        cps00.find_neighbours(0);
        *cps00.controller_mut().probability_mut(0) = 1.0;
        assert_eq!(cps00.decide_unit(0), Some(true));
        assert_eq!(cps00.neighbours().len(), 4);
        assert_eq!(cps00.neighbours(), vec![1, 2, 3, 4]);
        cps00.update_probabilities(-0.8);
        assert_delta!(cps00.controller().probability(1), 0.0, 1e-12);
        assert_delta!(cps00.controller().probability(2), 0.0, 1e-12);
        assert_delta!(cps00.controller().probability(3), 0.0, 1e-12);
        assert_delta!(cps00.controller().probability(4), 0.0, 1e-12);

        let s = CorrelatedPoissonSampling::draw(&RAND00, &prob, 1e-12);
        assert_eq!(s, vec![0, 5]);
    }

    #[test]
    fn lcps() {
        let (data, prob) = data_10_2();
        let mut cps00 = LocallyCorrelatedPoissonSampling::new(&RAND00, &prob, 1e-12, &data, 2);
        assert_eq!(cps00.draw_unit(), Some(8));
        *cps00.controller_mut().probability_mut(8) = 0.0;
        assert_eq!(cps00.decide_unit(8), Some(false));
        assert_eq!(cps00.draw_unit(), Some(2));
        *cps00.controller_mut().probability_mut(2) = 0.0;
        assert_eq!(cps00.decide_unit(2), Some(false));
        assert_eq!(cps00.draw_unit(), Some(5));
        *cps00.controller_mut().probability_mut(5) = 0.0;
        assert_eq!(cps00.decide_unit(5), Some(false));
        assert_eq!(cps00.draw_unit(), Some(4));
        *cps00.controller_mut().probability_mut(4) = 0.0;
        assert_eq!(cps00.decide_unit(4), Some(false));
        assert_eq!(cps00.draw_unit(), Some(3));
        assert_eq!(cps00.decide_unit(3), None);

        let s = LocallyCorrelatedPoissonSampling::draw(&RAND00, &prob, 1e-12, &data, 2);
        assert_eq!(s, vec![4, 8]);
    }
}
