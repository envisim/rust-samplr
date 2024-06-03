use crate::pivotal_method::base::{DrawVariant, PivotalMethod};
use envisim_samplr_utils::{uniform_random::RandomGenerator, Indices};

struct SequentialPivotalMethod {
    population_size: usize,
    pair: [usize; 2],
    idx: Indices,
}

pub fn sequential_pivotal_method(
    rand: &RandomGenerator,
    probabilities: &[f64],
    eps: f64,
) -> Vec<usize> {
    let population_size = probabilities.len();
    let mut pm = PivotalMethod::new(
        SequentialPivotalMethod {
            population_size: population_size,
            pair: [0, 1],
            idx: Indices::new_fill(population_size),
        },
        &rand,
        probabilities,
        eps,
    );

    pm.run_and_return()
}

impl DrawVariant for SequentialPivotalMethod {
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
    fn draw(&mut self, _: &RandomGenerator) -> Option<(usize, usize)> {
        if self.len() <= 1 {
            return None;
        }

        if !self.idx.includes(self.pair[0]) {
            self.pair[0] = self.pair[1];

            while !self.idx.includes(self.pair[0]) {
                self.pair[0] += 1;

                if self.pair[0] >= self.population_size {
                    panic!("spm looped past last unit");
                }
            }

            self.pair[1] = self.pair[0] + 1;
        }

        while !self.idx.includes(self.pair[1]) {
            self.pair[1] += 1;

            if self.pair[1] >= self.population_size {
                panic!("spm looped past last unit");
            }
        }

        return Some((self.pair[0], self.pair[1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAND00: RandomGenerator = || 0.0;
    const RAND99: RandomGenerator = || 0.999;

    #[test]
    fn spm() {
        let spm = sequential_pivotal_method(&RAND00, &vec![0.5, 0.5, 0.5, 0.5], 1e-12);

        let result: Vec<usize> = vec![1, 3];
        assert_eq!(spm, result);
    }

    #[test]
    fn spm2() {
        let spm = sequential_pivotal_method(&RAND99, &vec![0.5, 0.5, 0.5, 0.5], 1e-12);

        let result: Vec<usize> = vec![0, 2];
        assert_eq!(spm, result);
    }
}
