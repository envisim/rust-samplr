use envisim_samplr_utils::{
    container::Container,
    kd_tree::{midpoint_slide, Node, SearcherWeighted},
    matrix::{OperateMatrix, RefMatrix},
    random_generator::RandomList,
    utils::usize_to_f64,
};

pub trait CorrelatedPoissonVariant<'a, R>
where
    R: RandomList,
{
    fn select_unit(&mut self, container: &Container<'a, R>) -> Option<usize>;
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    );
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct CorrelatedPoissonSampler<'a, R, T>
where
    R: RandomList,
    T: CorrelatedPoissonVariant<'a, R>,
{
    container: Box<Container<'a, R>>,
    variant: Box<T>,
}

pub struct SequentialCorrelatedPoissonSampling {
    unit: usize,
}

pub struct SpatiallyCorrelatedPoissonSampling<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<SearcherWeighted>,
    unit: usize, // Sequential also, usize::MAX
}

pub struct LocallyCorrelatedPoissonSampling<'a> {
    scps: SpatiallyCorrelatedPoissonSampling<'a>,
    candidates: Vec<usize>,
}

impl SequentialCorrelatedPoissonSampling {
    pub fn new<'a, R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
    ) -> CorrelatedPoissonSampler<'a, R, Self>
    where
        R: RandomList,
    {
        CorrelatedPoissonSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(SequentialCorrelatedPoissonSampling { unit: 0 }),
        }
    }

    pub fn sample<'a, R>(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize>
    where
        R: RandomList,
    {
        Self::new(rand, probabilities, eps)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> SpatiallyCorrelatedPoissonSampling<'a> {
    pub fn new<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> CorrelatedPoissonSampler<'a, R, Self>
    where
        R: RandomList,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(SearcherWeighted::new(&tree));

        CorrelatedPoissonSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(SpatiallyCorrelatedPoissonSampling {
                tree: tree,
                searcher: searcher,
                unit: usize::MAX,
            }),
        }
    }

    pub fn sample<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomList,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }

    pub fn new_sequential<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> CorrelatedPoissonSampler<'a, R, Self>
    where
        R: RandomList,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(SearcherWeighted::new(&tree));

        CorrelatedPoissonSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(SpatiallyCorrelatedPoissonSampling {
                tree: tree,
                searcher: searcher,
                unit: 0,
            }),
        }
    }

    pub fn sample_sequential<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomList,
    {
        Self::new_sequential(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> LocallyCorrelatedPoissonSampling<'a> {
    pub fn new<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> CorrelatedPoissonSampler<'a, R, Self>
    where
        R: RandomList,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(SearcherWeighted::new(&tree));

        CorrelatedPoissonSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(LocallyCorrelatedPoissonSampling {
                scps: SpatiallyCorrelatedPoissonSampling {
                    tree: tree,
                    searcher: searcher,
                    unit: usize::MAX,
                },
                candidates: Vec::<usize>::with_capacity(20),
            }),
        }
    }

    pub fn sample<R>(
        rand: &'a R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomList,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a, R, T> CorrelatedPoissonSampler<'a, R, T>
where
    R: RandomList,
    T: CorrelatedPoissonVariant<'a, R>,
{
    #[inline]
    fn decide_selected(&mut self, id: usize) -> (f64, f64) {
        let probability = self.container.probabilities()[id];
        let mut quota = probability;

        if self.container.random().from_list(id) < probability {
            self.container.probabilities_mut()[id] = 1.0;
            quota -= 1.0;
        } else {
            self.container.probabilities_mut()[id] = 0.0;
        }

        self.variant.decide_unit(&mut self.container, id);

        (probability, quota)
    }

    pub fn sample(&mut self) -> &mut Self {
        while let Some(id) = self.variant.select_unit(&self.container) {
            let (probability, quota) = self.decide_selected(id);
            self.variant
                .update_neighbours(&mut self.container, id, probability, quota);
        }

        self
    }
    #[inline]
    pub fn get_sample(&mut self) -> &[usize] {
        self.container.sample().get()
    }
    #[inline]
    pub fn get_sorted_sample(&mut self) -> &[usize] {
        self.container.sample_mut().sort().get()
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for SequentialCorrelatedPoissonSampling
where
    R: RandomList,
{
    fn select_unit(&mut self, container: &Container<'a, R>) -> Option<usize> {
        if container.indices().len() == 0 {
            return None;
        }

        while self.unit < container.population_size() {
            if container.indices().contains(self.unit) {
                return Some(self.unit);
            }

            self.unit += 1;
        }

        panic!("unreachable code");
    }
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        let mut nid = id + 1;
        let mut remaining_weight: f64 = 1.0;

        while nid < container.population_size() && remaining_weight > 0.0 {
            if !container.indices().contains(nid) {
                nid += 1;
                continue;
            }

            let possible_weight = container.probabilities().weight_to(probability, nid);
            let weight = possible_weight.min(remaining_weight);
            container.probabilities_mut()[nid] += weight * quota;
            self.decide_unit(container, nid);
            remaining_weight -= possible_weight;
            nid += 1;
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id)
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for SpatiallyCorrelatedPoissonSampling<'a>
where
    R: RandomList,
{
    fn select_unit(&mut self, container: &Container<'a, R>) -> Option<usize> {
        if container.indices().len() <= 1 {
            return container.indices().first().cloned();
        }

        // Random order
        if self.unit == usize::MAX {
            return container.indices_random().cloned();
        }

        // Sequential order
        while self.unit < container.population_size() {
            if container.indices().contains(self.unit) {
                return Some(self.unit);
            }

            self.unit += 1;
        }

        panic!("unreachable code");
    }
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        if container.indices().len() == 0 {
            return;
        }

        self.searcher.find_neighbours_of_iter(
            &self.tree,
            container.probabilities(),
            self.tree.data().into_row_iter(id),
            probability,
        );

        let mut remaining_weight: f64 = 1.0;
        let mut i: usize = 0;

        while i < self.searcher.neighbours().len() {
            // Start by adding up the weights of all ties
            let mut sum_of_tie_weights = self.searcher.weight_k(i);
            let distance = self.searcher.distance_k(i);
            let mut j: usize = i + 1;

            while j < self.searcher.neighbours().len() && self.searcher.distance_k(j) == distance {
                sum_of_tie_weights += self.searcher.weight_k(j);
                j += 1;
            }

            // If the sum of all ties are less than the remaining weight, we can
            // continue as usual
            if sum_of_tie_weights < remaining_weight {
                while i < j {
                    let id = self.searcher.neighbours()[i];
                    let removable_weight = self.searcher.weight_k(i);
                    container.probabilities_mut()[id] += removable_weight * quota;
                    self.decide_unit(container, id);
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
            self.searcher.sort_by_weight(i, j);

            while i < j {
                let id = self.searcher.neighbours()[i];
                let removable_weight = self.searcher.weight_k(i).min(remaining_weight / sharers);
                container.probabilities_mut()[id] += removable_weight * quota;
                self.decide_unit(container, id);
                remaining_weight -= removable_weight;
                sharers -= 1.0;
                i += 1;
            }

            i = j;
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).and_then(|r| {
            self.tree.remove_unit(id);
            Some(r)
        })
    }
}

impl<'a, R> CorrelatedPoissonVariant<'a, R> for LocallyCorrelatedPoissonSampling<'a>
where
    R: RandomList,
{
    fn select_unit(&mut self, container: &Container<'a, R>) -> Option<usize> {
        if container.indices().len() <= 1 {
            return container.indices().first().cloned();
        } else if container.indices().len() == 2 {
            return container.indices_random().cloned();
        }

        let mut minimum_distance = f64::MAX;
        self.candidates.clear();

        // Loop through all remaining units
        let mut i = 0;
        while i < container.indices().len() {
            let id = *container.indices().get(i).unwrap();
            self.scps.searcher.find_neighbours_of_id(
                &self.scps.tree,
                container.probabilities(),
                id,
            );
            // We are guaranteed to have at least one neighbour by the
            // if's in the beginning
            let distance = self
                .scps
                .searcher
                .distance_k(self.scps.searcher.neighbours().len() - 1);

            if distance < minimum_distance {
                self.candidates.clear();
                self.candidates.push(id);
                minimum_distance = distance;
            } else if distance == minimum_distance {
                self.candidates.push(id);
            }

            i += 1;
        }

        container.random().rslice(&self.candidates).cloned()
    }
    #[inline]
    fn update_neighbours(
        &mut self,
        container: &mut Container<'a, R>,
        id: usize,
        probability: f64,
        quota: f64,
    ) {
        self.scps
            .update_neighbours(container, id, probability, quota)
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        self.scps.decide_unit(container, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_delta, data_10_2, EPS, RAND00, RAND99};

    #[test]
    fn cps_sampler() {
        let (_data, prob) = data_10_2();
        let mut cps = SequentialCorrelatedPoissonSampling::new(&RAND00, &prob, EPS);
        assert_eq!(cps.decide_selected(7), (0.2, -0.8));
        cps = SequentialCorrelatedPoissonSampling::new(&RAND99, &prob, EPS);
        assert_eq!(cps.decide_selected(7), (0.2, 0.2));
    }

    fn decide_and_update<'a, R, T>(
        cps: &mut CorrelatedPoissonSampler<'a, R, T>,
        id: usize,
    ) -> (usize, f64, f64)
    where
        R: envisim_samplr_utils::random_generator::RandomList,
        T: CorrelatedPoissonVariant<'a, R>,
    {
        let (p, q) = cps.decide_selected(id);
        cps.variant.update_neighbours(&mut cps.container, id, p, q);
        (id, p, q)
    }

    #[test]
    fn cps_variant() {
        type ST = SequentialCorrelatedPoissonSampling;
        let (_data, prob) = data_10_2();

        let mut cps = ST::new(&RAND00, &prob, EPS);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(0));
        decide_and_update(&mut cps, 0);
        assert_delta!(cps.container.probabilities()[1], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[3], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[4], 0.0, EPS);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(5));

        cps = ST::new(&RAND99, &prob, EPS);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(0));
        decide_and_update(&mut cps, 0);
        assert_delta!(cps.container.probabilities()[1], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[3], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[4], 0.25, EPS);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(1));

        let s0 = ST::sample(&RAND00, &prob, EPS);
        assert_eq!(s0, vec![0, 5]);
        let s1 = ST::sample(&RAND99, &prob, EPS);
        assert_eq!(s1, vec![4, 9]);
    }

    #[test]
    fn scps_variant() {
        type ST<'a> = SpatiallyCorrelatedPoissonSampling<'a>;
        let (data, prob) = data_10_2();

        let mut cps = ST::new(&RAND00, &prob, EPS, &data, 2);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(0));
        decide_and_update(&mut cps, 0);
        assert_delta!(cps.container.probabilities()[1], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[8], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[4], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.0, EPS);

        cps = ST::new(&RAND99, &prob, EPS, &data, 2);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(9));
        decide_and_update(&mut cps, 9);
        assert_delta!(cps.container.probabilities()[4], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[0], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[7], 0.25, EPS);

        let s0 = ST::sample(&RAND00, &prob, EPS, &data, 2);
        assert_eq!(s0, vec![0, 9]);
    }

    #[test]
    fn lcps_variant() {
        type ST<'a> = LocallyCorrelatedPoissonSampling<'a>;
        let (data, prob) = data_10_2();

        let mut cps = ST::new(&RAND00, &prob, EPS, &data, 2);

        assert_eq!(cps.variant.select_unit(&cps.container), Some(8));
        decide_and_update(&mut cps, 8);
        assert_delta!(cps.container.probabilities()[3], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[5], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.0, EPS);
        assert_delta!(cps.container.probabilities()[1], 0.0, EPS);

        cps = ST::new(&RAND99, &prob, EPS, &data, 2);
        assert_eq!(cps.variant.select_unit(&cps.container), Some(8));
        decide_and_update(&mut cps, 8);
        assert_delta!(cps.container.probabilities()[3], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[5], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[2], 0.25, EPS);
        assert_delta!(cps.container.probabilities()[1], 0.25, EPS);

        let s0 = ST::sample(&RAND00, &prob, EPS, &data, 2);
        assert_eq!(s0, vec![4, 8]);
    }
}
