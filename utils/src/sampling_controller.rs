use crate::generate_random::GenerateRandom;
use crate::indices::Indices;
use crate::matrix::{OperateMatrix, RefMatrix};

pub trait AccessBaseController<R: GenerateRandom> {
    fn random(&self) -> &R;
    #[inline]
    fn random_get<'t, T>(&self, slice: &'t [T]) -> Option<&'t T> {
        self.random().random_get(slice)
    }
    fn eps(&self) -> &f64;
    fn probabilities(&self) -> &[f64];
    unsafe fn probabilities_mut(&mut self) -> &mut Vec<f64>;
    #[inline]
    fn probability(&self, id: usize) -> &f64 {
        &self.probabilities()[id]
    }
    #[inline]
    fn probability_mut(&mut self, id: usize) -> &mut f64 {
        unsafe { &mut (self.probabilities_mut()[id]) }
    }
    #[inline]
    fn probability_is_zero(&self, id: usize) -> bool {
        *self.probability(id) <= *self.eps()
    }
    #[inline]
    fn probability_is_one(&self, id: usize) -> bool {
        *self.probability(id) >= 1.0 - *self.eps()
    }
    #[inline]
    fn weight(&self, id0: usize, id1: usize) -> f64 {
        let p0 = *self.probability(id0);
        let p1 = *self.probability(id1);

        if p0 + p1 <= 1.0 {
            return p1 / (1.0 - p0);
        } else {
            return (1.0 - p1) / p0;
        }
    }
    #[inline]
    fn population_size(&self) -> usize {
        self.probabilities().len()
    }
    fn indices(&self) -> &Indices;
    fn indices_mut(&mut self) -> &mut Indices;
    #[inline]
    fn get_random_index(&self) -> Option<&usize> {
        self.indices().get_random(self.random())
    }
    fn sample(&self) -> &[usize];
    unsafe fn sample_mut(&mut self) -> &mut Vec<usize>;
    #[inline]
    fn sample_sort(&mut self) {
        unsafe {
            self.sample_mut().sort();
        }
    }
    #[inline]
    fn add_to_sample(&mut self, id: usize) {
        unsafe {
            self.sample_mut().push(id);
        }
    }
    #[inline]
    fn decide_unit(&mut self, id: usize) -> Option<bool> {
        if self.probability_is_zero(id) {
            self.indices_mut().remove(id);
            return Some(false);
        } else if self.probability_is_one(id) {
            self.indices_mut().remove(id);
            self.add_to_sample(id);
            return Some(true);
        }

        None
    }
    #[inline]
    fn update_last_unit(&mut self) -> Option<usize> {
        let last = self.indices().get_last();
        if last.is_none() {
            return None;
        }

        let id = *last.unwrap();

        *self.probability_mut(id) = if self.random().random_float() < *self.probability(id) {
            1.0
        } else {
            0.0
        };

        return Some(id);
    }
}

pub trait AccessDataController<R: GenerateRandom>: AccessBaseController<R> {
    fn data(&self) -> &RefMatrix;
    #[inline]
    fn data_nrow(&self) -> usize {
        self.data().nrow()
    }
    #[inline]
    fn data_ncol(&self) -> usize {
        self.data().ncol()
    }
    #[inline]
    fn data_dim(&self) -> (usize, usize) {
        self.data().dim()
    }
    fn distance_between(&self, id1: usize, unit2: &[f64]) -> f64;
}

pub struct BaseController<'a, R: GenerateRandom> {
    random: &'a R,
    eps: f64,
    probabilities: Vec<f64>,
    indices: Indices,
    sample: Vec<usize>,
}

pub struct DataController<'a, R: GenerateRandom> {
    data: &'a RefMatrix<'a>,
    base: BaseController<'a, R>,
}

impl<'a, R: GenerateRandom> BaseController<'a, R> {
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64) -> Self {
        let population_size = probabilities.len();

        let mut controller = BaseController {
            random: rand,
            eps: eps,
            probabilities: probabilities.to_vec(),
            indices: Indices::new_fill(population_size),
            sample: Vec::<usize>::with_capacity(population_size),
        };

        for i in 0..population_size {
            controller.decide_unit(i);
        }

        controller
    }
}

impl<'a, R: GenerateRandom> DataController<'a, R> {
    pub fn new(rand: &'a R, probabilities: &[f64], eps: f64, data: &'a RefMatrix) -> Self {
        assert_eq!(data.nrow(), probabilities.len());

        DataController {
            data: data,
            base: BaseController::new(rand, probabilities, eps),
        }
    }
}

impl<'a, R: GenerateRandom> AccessBaseController<R> for BaseController<'a, R> {
    #[inline]
    fn random(&self) -> &R {
        self.random
    }
    #[inline]
    fn eps(&self) -> &f64 {
        &self.eps
    }
    #[inline]
    fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    #[inline]
    unsafe fn probabilities_mut(&mut self) -> &mut Vec<f64> {
        &mut self.probabilities
    }
    #[inline]
    fn indices(&self) -> &Indices {
        &self.indices
    }
    #[inline]
    fn indices_mut(&mut self) -> &mut Indices {
        &mut self.indices
    }
    #[inline]
    fn sample(&self) -> &[usize] {
        &self.sample
    }
    #[inline]
    unsafe fn sample_mut(&mut self) -> &mut Vec<usize> {
        &mut self.sample
    }
}

impl<'a, R: GenerateRandom> AccessBaseController<R> for DataController<'a, R> {
    #[inline]
    fn random(&self) -> &R {
        self.base.random()
    }
    #[inline]
    fn eps(&self) -> &f64 {
        self.base.eps()
    }
    #[inline]
    fn probabilities(&self) -> &[f64] {
        self.base.probabilities()
    }
    #[inline]
    unsafe fn probabilities_mut(&mut self) -> &mut Vec<f64> {
        self.base.probabilities_mut()
    }
    #[inline]
    fn indices(&self) -> &Indices {
        self.base.indices()
    }
    #[inline]
    fn indices_mut(&mut self) -> &mut Indices {
        self.base.indices_mut()
    }
    #[inline]
    fn sample(&self) -> &[usize] {
        self.base.sample()
    }
    #[inline]
    unsafe fn sample_mut(&mut self) -> &mut Vec<usize> {
        self.base.sample_mut()
    }
}

impl<'a, R: GenerateRandom> AccessDataController<R> for DataController<'a, R> {
    #[inline]
    fn data(&self) -> &RefMatrix {
        self.data
    }
    #[inline]
    fn distance_between(&self, id1: usize, unit2: &[f64]) -> f64 {
        assert!(self.data.ncol() == unit2.len());
        assert!(id1 < self.data.nrow());
        return self.data.distance_to_row(id1, unit2);
    }
}
