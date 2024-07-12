use crate::{indices::Indices, probability::Probabilities, random_generator::RandomGenerator};

pub struct Sample {
    sample: Vec<usize>,
}

impl Sample {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Sample {
            sample: Vec::<usize>::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn add(&mut self, idx: usize) {
        self.sample.push(idx);
    }

    #[inline]
    pub fn sort(&mut self) -> &mut Self {
        self.sample.sort_unstable();
        self
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.sample.to_vec()
    }
}

pub struct Container<'a, R>
where
    R: RandomGenerator,
{
    random: &'a R,
    probabilities: Probabilities,
    indices: Indices,
    sample: Sample,
}

impl<'a, R> Container<'a, R>
where
    R: RandomGenerator,
{
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64) -> Self {
        let population_size = probabilities.len();

        let mut container = Container {
            random: rand,
            probabilities: Probabilities::with_values(probabilities),
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };

        container.probabilities.eps = eps;

        for i in 0..population_size {
            container.decide_unit(i);
        }

        container
    }

    #[inline]
    pub fn random(&self) -> &R {
        self.random
    }

    #[inline]
    pub fn random_slice<'b, T>(&self, slice: &'b [T]) -> Option<&'b T> {
        self.random.rslice(slice)
    }

    #[inline]
    pub fn probabilities(&self) -> &Probabilities {
        &self.probabilities
    }

    #[inline]
    pub fn probabilities_mut(&mut self) -> &mut Probabilities {
        &mut self.probabilities
    }

    #[inline]
    pub fn indices(&self) -> &Indices {
        &self.indices
    }

    // #[inline]
    // pub fn indices_mut(&mut self) -> &Indices {
    //     &mut self.indices
    // }

    #[inline]
    pub fn indices_random(&self) -> Option<&usize> {
        self.indices.random(self.random)
    }

    #[inline]
    pub fn sample(&self) -> &Sample {
        &self.sample
    }

    #[inline]
    pub fn sample_mut(&mut self) -> &mut Sample {
        &mut self.sample
    }

    #[inline]
    pub fn population_size(&self) -> usize {
        self.probabilities.len()
    }

    #[inline]
    pub fn decide_unit(&mut self, idx: usize) -> Option<bool> {
        if self.probabilities.is_zero(idx) {
            self.indices.remove(idx);
            return Some(false);
        } else if self.probabilities.is_one(idx) {
            self.indices.remove(idx);
            self.sample.add(idx);
            return Some(true);
        }

        None
    }

    #[inline]
    pub fn update_last_unit(&mut self) -> Option<usize> {
        let last = self.indices.last();
        if last.is_none() {
            return None;
        }

        let id = *last.unwrap();

        self.probabilities[id] = if self.random.rf64() < self.probabilities[id] {
            1.0
        } else {
            0.0
        };

        return Some(id);
    }
}
