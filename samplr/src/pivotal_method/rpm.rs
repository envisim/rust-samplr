use crate::pivotal_method::base::{DrawVariant, PivotalMethod};
use envisim_samplr_utils::{
    uniform_random::{discrete_uniform_u, RandomGenerator},
    Indices,
};

struct RandomPivotalMethod {
    idx: Indices,
}

pub fn random_pivotal_method(
    rand: &RandomGenerator,
    probabilities: &[f64],
    eps: f64,
) -> Vec<usize> {
    let population_size = probabilities.len();
    let mut pm = PivotalMethod::new(
        RandomPivotalMethod {
            idx: Indices::new_fill(population_size),
        },
        &rand,
        probabilities,
        eps,
    );

    pm.run_and_return()
}

impl DrawVariant for RandomPivotalMethod {
    fn len(&self) -> usize {
        self.idx.len()
    }
    fn remove(&mut self, id: usize) -> &mut Self {
        self.idx.remove(id);
        self
    }
    fn draw_last(&mut self) -> Option<usize> {
        if self.len() == 1 {
            return Some(self.idx.get_first());
        }

        None
    }
    fn draw(&mut self, rand: &RandomGenerator) -> Option<(usize, usize)> {
        let len = self.len();
        if len <= 1 {
            return None;
        }

        let id1 = self.idx.get_random(rand);
        let mut id2 = self.idx.get_at(discrete_uniform_u(rand, len - 1));

        if id1 == id2 {
            id2 = self.idx.get_last();
        }

        return Some((id1, id2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAND00: RandomGenerator = || 0.0;
    const RAND99: RandomGenerator = || 0.999;

    #[test]
    fn rpm() {
        let rpm = random_pivotal_method(&RAND00, &vec![0.5, 0.5, 0.5, 0.5], 1e-12);

        let result: Vec<usize> = vec![1, 3];
        assert_eq!(rpm, result);
    }

    #[test]
    fn rpm2() {
        let rpm = random_pivotal_method(&RAND99, &vec![0.5, 0.5, 0.5, 0.5], 1e-12);

        let result: Vec<usize> = vec![1, 3];
        assert_eq!(rpm, result);
    }
}
