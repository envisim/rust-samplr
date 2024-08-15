use crate::utils::Container;
use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::kd_tree::{midpoint_slide, Node, Searcher};
use envisim_utils::matrix::RefMatrix;
use envisim_utils::utils::{random_element, sum, usize_to_f64};
use rand::Rng;
use rustc_hash::FxHashSet;
use std::num::NonZeroUsize;

type Pair = (usize, usize);

pub trait PivotalMethodVariant<'a, R>
where
    R: Rng + ?Sized,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)>;
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool>;
}

pub struct PivotalMethodSampler<'a, R, T>
where
    R: Rng + ?Sized,
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
    ) -> Result<PivotalMethodSampler<'a, R, Self>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Ok(PivotalMethodSampler {
            container: Box::new(Container::new(rand, probabilities, eps)?),
            variant: Box::new(Self { pair: (0, 1) }),
        })
    }

    pub fn sample<R>(
        rand: &mut R,
        probabilities: &[f64],
        eps: f64,
    ) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Self::new(rand, probabilities, eps).map(|mut s| s.sample().get_sorted_sample().to_vec())
    }
}

impl RandomPivotalMethod {
    pub fn new<'a, R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
    ) -> Result<PivotalMethodSampler<'a, R, Self>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Ok(PivotalMethodSampler {
            container: Box::new(Container::new(rand, probabilities, eps)?),
            variant: Box::new(Self {}),
        })
    }

    #[inline]
    pub fn sample<R>(
        rand: &mut R,
        probabilities: &[f64],
        eps: f64,
    ) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Self::new(rand, probabilities, eps).map(|mut s| s.sample().get_sorted_sample().to_vec())
    }
}

impl<'a> LocalPivotalMethod1<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<PivotalMethodSampler<'a, R, Self>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        let container = Box::new(Container::new(rand, probabilities, eps)?);
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        )?);
        let searcher = Box::new(Searcher::new(&tree, 1)?);

        Ok(PivotalMethodSampler {
            container,
            variant: Box::new(LocalPivotalMethod1 {
                tree,
                searcher,
                candidates: Vec::<usize>::with_capacity(20),
            }),
        })
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .map(|mut s| s.sample().get_sorted_sample().to_vec())
    }
}

impl<'a> LocalPivotalMethod1S<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<PivotalMethodSampler<'a, R, Self>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        let container = Box::new(Container::new(rand, probabilities, eps)?);
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        )?);
        let searcher = Box::new(Searcher::new(&tree, 1)?);
        let remaining_units = container.indices().len();

        Ok(PivotalMethodSampler {
            container,
            variant: Box::new(LocalPivotalMethod1S {
                tree,
                searcher,
                candidates: Vec::<usize>::with_capacity(20),
                history: Vec::<usize>::with_capacity(remaining_units),
            }),
        })
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .map(|mut s| s.sample().get_sorted_sample().to_vec())
    }
}

impl<'a> LocalPivotalMethod2<'a> {
    pub fn new<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<PivotalMethodSampler<'a, R, Self>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        let container = Box::new(Container::new(rand, probabilities, eps)?);
        let tree = Box::new(Node::new_from_indices(
            midpoint_slide,
            bucket_size,
            data,
            container.indices(),
        )?);
        let searcher = Box::new(Searcher::new(&tree, 1)?);

        Ok(PivotalMethodSampler {
            container,
            variant: Box::new(LocalPivotalMethod2 { tree, searcher }),
        })
    }

    #[inline]
    pub fn sample<R>(
        rand: &'a mut R,
        probabilities: &[f64],
        eps: f64,
        data: &'a RefMatrix,
        bucket_size: NonZeroUsize,
    ) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        Self::new(rand, probabilities, eps, data, bucket_size)
            .map(|mut s| s.sample().get_sorted_sample().to_vec())
    }
}

impl<'a, R, T> PivotalMethodSampler<'a, R, T>
where
    R: Rng + ?Sized,
    T: PivotalMethodVariant<'a, R>,
{
    pub fn sample(&mut self) -> &mut Self {
        while let Some(units) = self.variant.select_units(&mut self.container) {
            let rv = self.container.rng().gen::<f64>();
            self.update_probabilities(units, rv);
        }

        if let Some(id) = self.container.update_last_unit() {
            self.variant.decide_unit(&mut self.container, id);
        }

        self
    }
    #[inline]
    fn update_probabilities(&mut self, (id1, id2): Pair, rv: f64) {
        let mut p1 = self.container.probabilities()[id1];
        let mut p2 = self.container.probabilities()[id2];
        let psum = p1 + p2;

        if psum > 1.0 {
            if 1.0 - p2 > rv * (2.0 - psum) {
                p1 = 1.0;
                p2 = psum - 1.0;
            } else {
                p1 = psum - 1.0;
                p2 = 1.0;
            }
        } else if p2 > rv * psum {
            // psum <= 1.0
            p1 = 0.0;
            p2 = psum;
        } else {
            // psum <= 1.0
            p1 = psum;
            p2 = 0.0;
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
    R: Rng + ?Sized,
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
        container.decide_unit(id).unwrap()
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for RandomPivotalMethod
where
    R: Rng + ?Sized,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_draw().unwrap();
        let k = container.rng().gen_range(0..(len - 1));
        let mut id2 = *container.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = *container.indices().last().unwrap();
        }

        Some((id1, id2))
    }
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap()
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1<'a>
where
    R: Rng + ?Sized,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        loop {
            let id1 = *container.indices_draw().unwrap();
            self.searcher
                .find_neighbours_of_id(&self.tree, id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;

            while i < self.candidates.len() {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if !self.candidates.is_empty() {
                let id2 = *random_element(container.rng(), &self.candidates).unwrap();
                return Some((id1, id2));
            }
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap().map(|r| {
            self.tree.remove_unit(id).unwrap();
            r
        })
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod1S<'a>
where
    R: Rng + ?Sized,
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

        if self.history.is_empty() {
            self.history.push(*container.indices_draw().unwrap());
        }

        loop {
            let id1 = *self.history.last().unwrap();
            self.searcher
                .find_neighbours_of_id(&self.tree, id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;
            let mut len = self.candidates.len();

            while i < len {
                self.searcher
                    .find_neighbours_of_id(&self.tree, self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    // If we does not find any compatible matches, we use the candidates to continue our seach
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2 = *random_element(container.rng(), &self.candidates[0..len]).unwrap();
                return Some((id1, id2));
            }

            if self.history.len() == container.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history
                .push(*random_element(container.rng(), &self.candidates).unwrap());
        }
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap().map(|r| {
            self.tree.remove_unit(id).unwrap();
            r
        })
    }
}

impl<'a, R> PivotalMethodVariant<'a, R> for LocalPivotalMethod2<'a>
where
    R: Rng + ?Sized,
{
    fn select_units(&mut self, container: &mut Container<'a, R>) -> Option<(usize, usize)> {
        let len = container.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((container.indices().list()[0], container.indices().list()[1]));
        }

        let id1 = *container.indices_draw().unwrap();
        self.searcher
            .find_neighbours_of_id(&self.tree, id1)
            .unwrap();
        let id2 = *random_element(container.rng(), self.searcher.neighbours()).unwrap();

        Some((id1, id2))
    }
    #[inline]
    fn decide_unit(&mut self, container: &mut Container<'a, R>, id: usize) -> Option<bool> {
        container.decide_unit(id).unwrap().map(|r| {
            self.tree.remove_unit(id).unwrap();
            r
        })
    }
}

pub fn hierarchical_local_pivotal_method_2<'a, R>(
    rand: &'a mut R,
    probabilities: &[f64],
    eps: f64,
    data: &'a RefMatrix,
    bucket_size: NonZeroUsize,
    sizes: &[usize],
) -> Result<Vec<Vec<usize>>, SamplingError>
where
    R: Rng + ?Sized,
{
    {
        let psum = sum(probabilities);
        let sizesum = usize_to_f64(sizes.iter().sum());
        if !(psum - eps..psum + eps).contains(&sizesum) {
            return Err(SamplingError::from(InputError::NotInteger));
        }
    }
    InputError::check_empty(sizes)?;

    if sizes.len() == 1 {
        return Ok(vec![LocalPivotalMethod2::sample(
            rand,
            probabilities,
            eps,
            data,
            bucket_size,
        )?]);
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());
    let mut pm = LocalPivotalMethod2::new(rand, probabilities, eps, data, bucket_size)?;

    let mut main_sample: FxHashSet<usize> = pm.sample().get_sample().iter().cloned().collect();

    for (i, &size) in sizes[0..sizes.len() - 1].iter().enumerate() {
        assert!(pm.container.indices().is_empty());

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.container.sample_mut().clear();

        let prob = usize_to_f64(size) / usize_to_f64(main_sample.len());

        // Reset probs and add to indices/tree
        for id in 0..pm.container.population_size() {
            if main_sample.contains(&id) {
                pm.container.probabilities_mut()[id] = prob;
                pm.container.indices_mut().insert(id).unwrap();
                pm.variant.tree.insert_unit(id).unwrap();
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

    Ok(return_sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_test_utils::*;

    #[test]
    fn update_probabilities() {
        let mut rng = seeded_rng();
        let mut pm = SequentialPivotalMethod::new(&mut rng, &PROB_10_E, EPS).unwrap();
        pm.update_probabilities((0, 1), 0.0);
        assert_delta!(pm.container.probabilities()[0], 0.0);
        assert_delta!(pm.container.probabilities()[1], 0.4);
        assert!(!pm.container.indices().contains(0));
        assert!(pm.container.indices().contains(1));
        pm.update_probabilities((2, 1), 0.0);
        pm.update_probabilities((3, 1), 0.0);
        pm.update_probabilities((4, 1), 0.0);
        assert!(!pm.container.indices().contains(1));
        assert!(pm.container.sample().get().contains(&1));
    }
}
