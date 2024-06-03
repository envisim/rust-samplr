use envisim_samplr_utils::{
    uniform_random::{uniform_max, uniform_std, RandomGenerator},
    Probability,
};

pub trait DrawVariant {
    fn len(&self) -> usize;
    fn remove(&mut self, id: usize) -> &mut Self;
    fn draw_last(&mut self) -> Option<usize>;
    fn draw(&mut self, rand: &RandomGenerator) -> Option<(usize, usize)>;
}

pub struct PivotalMethod<'a, T: DrawVariant> {
    pub rand: &'a RandomGenerator,
    eps: f64,
    pub probabilities: Vec<f64>,
    sample: Vec<usize>,
    variant: T,
}

impl<'a, T: DrawVariant> PivotalMethod<'a, T> {
    pub fn new(variant: T, rand: &'a RandomGenerator, probabilities: &[f64], eps: f64) -> Self {
        let n = probabilities.len();

        let mut pm = PivotalMethod {
            rand: rand,
            eps: eps,
            probabilities: Vec::<f64>::with_capacity(n),
            sample: Vec::<usize>::with_capacity(n),
            variant: variant,
        };

        for i in 0..n {
            let p = probabilities[i];

            if p.is_zero(eps) {
                pm.probabilities.push(0.0);
                pm.variant.remove(i);
                pm.sample.push(i);
                continue;
            } else if p.is_one(eps) {
                pm.probabilities.push(1.0);
                pm.variant.remove(i);
                continue;
            }

            pm.probabilities.push(p);
        }

        pm
    }

    pub fn get_sample(&self) -> &[usize] {
        &self.sample
    }

    fn decide_unit(&mut self, id: usize) -> &mut Self {
        if self.probabilities[id].is_one(self.eps) {
            self.sample.push(id);
            self.variant.remove(id);
        } else if self.probabilities[id].is_zero(self.eps) {
            self.variant.remove(id);
        }

        self
    }

    fn decide_trailing_unit(&mut self) -> &mut Self {
        if let Some(id) = self.variant.draw_last() {
            self.probabilities[id] = if uniform_std(&self.rand) < self.probabilities[id] {
                1.0
            } else {
                0.0
            };

            self.decide_unit(id);
        }

        self
    }

    pub fn run(&mut self) -> &mut Self {
        while let Some((id1, id2)) = self.variant.draw(&self.rand) {
            self.update_probabilities(id1, id2)
                .decide_unit(id1)
                .decide_unit(id2);
        }

        self.decide_trailing_unit()
    }

    pub fn run_and_return(&mut self) -> Vec<usize> {
        let mut sample = self.run().get_sample().to_vec();
        sample.sort();
        sample
    }

    fn update_probabilities(&mut self, id1: usize, id2: usize) -> &mut Self {
        let mut p1 = self.probabilities[id1];
        let mut p2 = self.probabilities[id2];
        let psum = p1 + p2;

        if psum > 1.0 {
            if 1.0 - p2 > uniform_max(&self.rand, 2.0 - psum) {
                p1 = 1.0;
                p2 = psum - 1.0;
            } else {
                p1 = psum - 1.0;
                p2 = 1.0;
            }
        } else {
            if p2 > uniform_max(&self.rand, psum) {
                p1 = 0.0;
                p2 = psum;
            } else {
                p1 = psum;
                p2 = 0.0;
            }
        }

        self.probabilities[id1] = p1;
        self.probabilities[id2] = p2;
        self
    }
}
