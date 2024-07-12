use envisim_samplr_utils::{
    container::Container,
    kd_tree::{midpoint_slide, Node, Searcher},
    matrix::RefMatrix,
    random_generator::RandomGenerator,
};

type Pair = (usize, usize);

pub trait PivotalMethodVariant<'a, R>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)>;
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
        rand: &'a R,
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

    pub fn sample<'a, R>(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize>
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
        rand: &'a R,
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

    pub fn sample<'a, R>(rand: &'a R, probabilities: &[f64], eps: f64) -> Vec<usize>
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
        rand: &'a R,
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

    pub fn sample<R>(
        rand: &'a R,
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
        rand: &'a R,
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

    pub fn sample<R>(
        rand: &'a R,
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
        rand: &'a R,
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

    pub fn sample<R>(
        rand: &'a R,
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
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)> {
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
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_random().unwrap();
        let mut id2 = *container
            .indices()
            .get(container.random().rusize(len - 1))
            .unwrap();

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
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)> {
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
        match container.decide_unit(id) {
            Some(inc_exc) => {
                self.tree.remove_unit(id);
                Some(inc_exc)
            }
            _ => None,
        }
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1S<'a>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)> {
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
        match container.decide_unit(id) {
            Some(inc_exc) => {
                self.tree.remove_unit(id);
                Some(inc_exc)
            }
            _ => None,
        }
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod2<'a>
where
    R: RandomGenerator,
{
    fn select_units(&mut self, container: &Container<'a, R>) -> Option<(usize, usize)> {
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
        match container.decide_unit(id) {
            Some(inc_exc) => {
                self.tree.remove_unit(id);
                Some(inc_exc)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_samplr_utils::random_generator::Constant;

    const RAND00: Constant = Constant::new(0.0);
    const RAND99: Constant = Constant::new(0.999);

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
        assert_eq!(lpm00.variant.select_units(&lpm00.container), Some((0, 1)));

        let mut lpm99 = LocalPivotalMethod1::new(&RAND99, &prob, 1e-12, &data, 2);
        assert_eq!(lpm99.variant.select_units(&lpm99.container), Some((9, 4)));
    }

    #[test]
    fn lpm1s() {
        let (data, prob) = data_10_2();

        let mut lpm00 = LocalPivotalMethod1S::new(&RAND99, &prob, 1e-12, &data, 2);
        let mut units: (usize, usize);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (9, 4));
        assert_eq!(lpm00.variant.history, vec![9]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (3, 5));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (3, 8));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (3, 6));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (2, 7));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (2, 3));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (2, 9));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3, 2]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        assert_eq!(units, (1, 0));
        assert_eq!(lpm00.variant.history, vec![9, 2, 8, 3, 2, 1]);
        lpm00.update_probabilities(units);
        units = lpm00.variant.select_units(&lpm00.container).unwrap();
        lpm00.update_probabilities(units);

        assert_eq!(lpm00.container.indices().len(), 0);
        assert_eq!(lpm00.get_sorted_sample(), vec![1, 3]);
    }

    #[test]
    fn lpm2() {
        let (data, prob) = data_10_2();

        let lpm00_result = LocalPivotalMethod2::sample(&RAND00, &prob, 1e-12, &data, 5);
        assert_eq!(lpm00_result, vec![1, 3]);
    }

    #[test]
    fn rpm() {
        let rpm00_result = RandomPivotalMethod::sample(&RAND00, &[0.5f64; 4], 1e-12);
        assert_eq!(rpm00_result, vec![1, 3]);

        let rpm99_result = RandomPivotalMethod::sample(&RAND99, &[0.5f64; 4], 1e-12);
        assert_eq!(rpm99_result, vec![0, 3]);
    }

    #[test]
    fn spm() {
        let spm00_result = SequentialPivotalMethod::sample(&RAND00, &[0.5f64; 4], 1e-12);
        assert_eq!(spm00_result, vec![1, 3]);

        let spm99_result = SequentialPivotalMethod::sample(&RAND99, &[0.5f64; 4], 1e-12);
        assert_eq!(spm99_result, vec![0, 2]);
    }
}
