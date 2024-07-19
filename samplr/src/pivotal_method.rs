use crate::macros::assert_delta;
use envisim_samplr_utils::{
    container::Container,
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::RefMatrix,
    random_generator::RandomGenerator,
    utils::usize_to_f64,
};
use rustc_hash::FxHashSet;

type Pair = (usize, usize);

pub trait PivotalMethodVariant<'a, R>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)>;
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct PivotalMethodSampler<'a, R, T>
where
    R: RandomGenerator,
    T: PivotalMethodVariant<'a, R>,
{
    container: Box<Container<'a, R>>,
    variant: Box<T>,
}

pub struct SequentialPivotalMethod {
    pair: Pair,
}
pub struct RandomPivotalMethod {}
pub struct LocalPivotalMethod1<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
    candidates: Vec<usize>,
}
pub struct LocalPivotalMethod1S<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
    candidates: Vec<usize>,
    history: Vec<usize>,
}
pub struct LocalPivotalMethod2<'a> {
    tree: Box<Node<'a>>,
    searcher: Box<Searcher>,
}

impl SequentialPivotalMethod {
    pub fn new<'a, R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
    ) -> PivotalMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        PivotalMethodSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(Self { pair: (0, 1) }),
        }
    }

    pub fn sample<'a, R>(rand: &'a mut R, probabilities: &[f64], eps: f64) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl RandomPivotalMethod {
    pub fn new<'a, R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
    ) -> PivotalMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        PivotalMethodSampler {
            container: Box::new(Container::new(rand, probabilities, eps)),
            variant: Box::new(Self {}),
        }
    }

    #[inline]
    pub fn sample<'a, R>(rand: &'a mut R, probabilities: &[f64], eps: f64) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> LocalPivotalMethod1<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> PivotalMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(Searcher::new(&tree, 1));

        PivotalMethodSampler {
            container: container,
            variant: Box::new(LocalPivotalMethod1 {
                tree: tree,
                searcher: searcher,
                candidates: Vec::<usize>::with_capacity(20),
            }),
        }
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> LocalPivotalMethod1S<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> PivotalMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(Searcher::new(&tree, 1));
        let remaining_units = container.indices().len();

        PivotalMethodSampler {
            container: container,
            variant: Box::new(LocalPivotalMethod1S {
                tree: tree,
                searcher: searcher,
                candidates: Vec::<usize>::with_capacity(20),
                history: Vec::<usize>::with_capacity(remaining_units),
            }),
        }
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a> LocalPivotalMethod2<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> PivotalMethodSampler<'a, R, Self>
    where
        R: RandomGenerator,
    {
        let container = Box::new(Container::new(rand, probabilities, eps));
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        ));
        let searcher = Box::new(Searcher::new(&tree, 1));

        PivotalMethodSampler {
            container: container,
            variant: Box::new(LocalPivotalMethod2 {
                tree: tree,
                searcher: searcher,
            }),
        }
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: usize,
    ) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .sample()
            .get_sorted_sample()
            .to_vec()
    }
}

impl<'a, R, T> PivotalMethodSampler<'a, R, T>
where
    R: RandomGenerator,
    T: PivotalMethodVariant<'a, R>,
{
    pub fn sample(&mut self) -> &mut Self {
        while let Some(units) = self.variant.select_units(&mut self.container) {
            self.update_probabilities(units);
        }

        if let Some(id) = self.container.update_last_unit() {
            self.variant.decide_unit(&mut self.container, id);
        }

        self
    }
    fn update_probabilities(&mut self, (id1, id2): Pair) {
        let mut p1 = self.container.probabilities()[id1];
        let mut p2 = self.container.probabilities()[id2];
        let psum = p1 + p2;

        if psum > 1.0 {
            if 1.0 - p2 > self.container.random().rf64_scaled(2.0 - psum) {
                p1 = 1.0;
                p2 = psum - 1.0;
            } else {
                p1 = psum - 1.0;
                p2 = 1.0;
            }
        } else {
            if p2 > self.container.random().rf64_scaled(psum) {
                p1 = 0.0;
                p2 = psum;
            } else {
                p1 = psum;
                p2 = 0.0;
            }
        }

        self.container.probabilities_mut()[id1] = p1;
        self.container.probabilities_mut()[id2] = p2;
        self.variant.decide_unit(&mut self.container, id1);
        self.variant.decide_unit(&mut self.container, id2);
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

impl<'a, R> PivotalMethodVariant<'a, R> for SequentialPivotalMethod
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        if container.indices().len() <= 1 {
            return None;
        }

        if !container.indices().contains(self.pair.0) {
            self.pair.0 = self.pair.1;

            while !container.indices().contains(self.pair.0) {
                self.pair.0 += 1;

                if self.pair.0 >= container.population_size() {
                    panic!("spm looped past last unit");
                }
            }

            self.pair.1 = self.pair.0 + 1;
        }

        while !container.indices().contains(self.pair.1) {
            self.pair.1 += 1;

            if self.pair.1 >= container.population_size() {
                panic!("spm looped past last unit");
            }
        }

        Some(self.pair)
    }
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id)
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for RandomPivotalMethod
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_random().unwrap();
        let k = container.random().rusize(len - 1);
        let mut id2 = *container.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = *container.indices().last().unwrap();
        }

        Some((id1, id2))
    }
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id)
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1<'a>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        loop {
            let id1 = *container.indices_random().unwrap();
            self.searcher.find_neighbours_of_id(&self.tree, id1);
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;

            while i < self.candidates.len() {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i]);

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if self.candidates.len() > 0 {
                let id2 = *container.random().rslice(&self.candidates).unwrap();
                return Some((id1, id2));
            }
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

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1S<'a>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        while let Some(&id) = self.history.last() {
            if container.indices().contains(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.len() == 0 {
            self.history.push(*container.indices_random().unwrap());
        }

        loop {
            let id1 = *self.history.last().unwrap();
            self.searcher.find_neighbours_of_id(&self.tree, id1);
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;
            let mut len = self.candidates.len();

            while i < len {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i]);

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    // If we does not find any compatible matches, we use the candidates to continue our seach
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2: usize = *container.random().rslice(&self.candidates[0..len]).unwrap();
                return Some((id1, id2));
            }

            if self.history.len() == container.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history
                .push(*container.random().rslice(&self.candidates).unwrap());
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

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod2<'a>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_random().unwrap();
        self.searcher.find_neighbours_of_id(&self.tree, id1);
        let id2 = *container
            .random()
            .rslice(self.searcher.neighbours())
            .unwrap();

        Some((id1, id2))
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).and_then(|r| {
            self.tree.remove_unit(id);
            Some(r)
        })
    }
}

pub fn hierarchical_local_pivotal_method_2<'a, R>(
    rand: &'a mut R,
    probabilities: &[f64],
    eps: f64,
    data: &'a RefMatrix,
    bucket_size: usize,
    sizes: &[usize],
) -> Vec<Vec<usize>>
where
    R: RandomGenerator,
{
    assert_delta!(
        probabilities.iter().sum::<f64>(),
        usize_to_f64(sizes.iter().sum()),
        eps
    );
    assert!(sizes.len() > 0);

    if sizes.len() == 1 {
        return vec![LocalPivotalMethod2::sample(
            rand,
            probabilities,
            eps,
            data,
            bucket_size,
        )];
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());
    let mut pm = LocalPivotalMethod2::new(rand, probabilities, eps, data, bucket_size);

    let mut main_sample: FxHashSet<usize> = pm.sample().get_sample().iter().cloned().collect();

    for (i, &size) in sizes[0..sizes.len() - 1].iter().enumerate() {
        assert_eq!(pm.container.indices().len(), 0);

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.container.sample_mut().clear();

        let prob = usize_to_f64(size) / usize_to_f64(main_sample.len());

        // Reset probs and add to indices/tree
        for id in 0..pm.container.population_size() {
            if main_sample.contains(&id) {
                pm.container.probabilities_mut()[id] = prob;
                pm.container.indices_mut().insert(id);
                pm.variant.tree.insert_unit(id);
            } else {
                pm.container.probabilities_mut()[id] = 0.0;
            }
        }

        pm.sample();
        return_sample.push(Vec::<usize>::with_capacity(pm.get_sample().len()));

        for &id in pm.get_sorted_sample().iter() {
            return_sample[i].push(id);
            main_sample.remove(&id);
        }
    }

    return_sample.push(main_sample.into_iter().collect());

    for s in return_sample.iter_mut() {
        s.sort_unstable();
    }

    return_sample
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_delta, data_10_2, data_20_2, gen_rand, EPS};

    fn select_and_update<'a, R, T>(pm: &mut PivotalMethodSampler<'a, R, T>) -> Pair
    where
        R: envisim_samplr_utils::random_generator::RandomGenerator,
        T: PivotalMethodVariant<'a, R>,
    {
        let units = pm.variant.select_units(&mut pm.container).unwrap();
        pm.update_probabilities(units);
        units
    }

    #[test]
    fn spm_variant() {
        type ST = SequentialPivotalMethod;
        let (mut rand00, mut rand99) = gen_rand();
        let (_data, prob) = data_10_2();

        let mut pm = ST::new(&mut rand00, &prob, EPS);
        assert_eq!(select_and_update(&mut pm), (0, 1));
        assert_delta!(pm.container.probabilities()[0], 0.0, EPS);
        assert_delta!(pm.container.probabilities()[1], 0.4, EPS);
        assert_eq!(select_and_update(&mut pm), (1, 2));

        pm = ST::new(&mut rand99, &prob, EPS);
        assert_eq!(select_and_update(&mut pm), (0, 1));
        assert_delta!(pm.container.probabilities()[0], 0.4, EPS);
        assert_delta!(pm.container.probabilities()[1], 0.0, EPS);
        assert_eq!(select_and_update(&mut pm), (0, 2));

        let s0 = ST::sample(&mut rand00, &prob, EPS);
        assert_eq!(s0, vec![4, 9]);
        let s1 = ST::sample(&mut rand99, &prob, EPS);
        assert_eq!(s1, vec![0, 5]);
    }

    #[test]
    fn rpm_variant() {
        type ST = RandomPivotalMethod;
        let (mut rand00, mut rand99) = gen_rand();
        let (_data, prob) = data_10_2();

        let mut pm = ST::new(&mut rand00, &prob, EPS);
        assert_eq!(select_and_update(&mut pm), (0, 9));
        assert_delta!(pm.container.probabilities()[0], 0.0, EPS);
        assert_delta!(pm.container.probabilities()[9], 0.4, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 8));

        pm = ST::new(&mut rand99, &prob, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 8));
        assert_delta!(pm.container.probabilities()[9], 0.4, EPS);
        assert_delta!(pm.container.probabilities()[8], 0.0, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 7));

        let s0 = ST::sample(&mut rand00, &prob, EPS);
        assert_eq!(s0, vec![1, 6]);
        let s1 = ST::sample(&mut rand99, &prob, EPS);
        assert_eq!(s1, vec![0, 9]);
    }

    #[test]
    fn lpm1_variant() {
        type ST<'a> = LocalPivotalMethod1<'a>;
        let (mut rand00, mut rand99) = gen_rand();
        let (data, prob) = data_10_2();

        let mut pm = ST::new(&mut rand00, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (0, 1));
        assert_delta!(pm.container.probabilities()[0], 0.0, EPS);
        assert_delta!(pm.container.probabilities()[1], 0.4, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 4));

        pm = ST::new(&mut rand99, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (9, 4));
        assert_delta!(pm.container.probabilities()[9], 0.4, EPS);
        assert_delta!(pm.container.probabilities()[4], 0.0, EPS);
        // assert_eq!(select_and_update(&mut pm), (9, 7));

        // let s0 = ST::sample(&mut rand00, &prob, EPS, &data, 2);
        // assert_eq!(s0, vec![1, 6]);
        // let s1 = ST::sample(&mut rand99, &prob, EPS, &data, 2);
        // assert_eq!(s1, vec![0, 9]);
    }

    #[test]
    fn lpm1s_variant() {
        type ST<'a> = LocalPivotalMethod1S<'a>;
        let (mut rand00, mut rand99) = gen_rand();
        let (data, prob) = data_10_2();

        let mut pm = ST::new(&mut rand00, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (0, 1));
        assert_delta!(pm.container.probabilities()[0], 0.0, EPS);
        assert_delta!(pm.container.probabilities()[1], 0.4, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 4));

        pm = ST::new(&mut rand99, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (9, 4));
        assert_delta!(pm.container.probabilities()[9], 0.4, EPS);
        assert_delta!(pm.container.probabilities()[4], 0.0, EPS);
        assert_eq!(select_and_update(&mut pm), (3, 5));

        let s1 = ST::sample(&mut rand99, &prob, EPS, &data, 2);
        assert_eq!(s1, vec![1, 3]);
    }

    #[test]
    fn lpm2_variant() {
        type ST<'a> = LocalPivotalMethod2<'a>;
        let (mut rand00, mut rand99) = gen_rand();
        let (data, prob) = data_10_2();

        let mut pm = ST::new(&mut rand00, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (0, 1));
        assert_delta!(pm.container.probabilities()[0], 0.0, EPS);
        assert_delta!(pm.container.probabilities()[1], 0.4, EPS);
        assert_eq!(select_and_update(&mut pm), (9, 4));

        pm = ST::new(&mut rand99, &prob, EPS, &data, 2);
        assert_eq!(select_and_update(&mut pm), (9, 4));
        assert_delta!(pm.container.probabilities()[9], 0.4, EPS);
        assert_delta!(pm.container.probabilities()[4], 0.0, EPS);
        assert_eq!(select_and_update(&mut pm), (8, 3));

        let s0 = ST::sample(&mut rand00, &prob, EPS, &data, 2);
        assert_eq!(s0, vec![1, 3]);
        let s1 = ST::sample(&mut rand99, &prob, EPS, &data, 2);
        // Depending on over/under 1.0 in last
        assert!(s1 == vec![8, 9] || s1 == vec![6, 8]);
    }

    #[test]
    fn hlpm2() {
        {
            let (mut rand00, mut _rand99) = gen_rand();
            let (data, prob) = data_10_2();
            let s = hierarchical_local_pivotal_method_2(&mut rand00, &prob, EPS, &data, 2, &[1, 1]);
            assert_eq!(s.len(), 2);
            assert_eq!(s[0].len(), 1);
            assert_eq!(s[1].len(), 1);
        }

        {
            let (mut rand00, mut _rand99) = gen_rand();
            let (data, prob) = data_20_2();
            let s =
                hierarchical_local_pivotal_method_2(&mut rand00, &prob, EPS, &data, 2, &[1, 2, 1]);
            assert_eq!(s.len(), 3);
            assert_eq!(s[0].len(), 1);
            assert_eq!(s[1].len(), 2);
            assert_eq!(s[2].len(), 1);
        }
    }
}
