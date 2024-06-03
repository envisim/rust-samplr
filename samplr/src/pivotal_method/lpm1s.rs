use crate::pivotal_method::base::{DrawVariant, PivotalMethod};
use envisim_samplr_utils::{
    kd_tree::{midpoint_slide, Tree},
    matrix::Matrix,
    uniform_random::{DrawRandom, RandomGenerator},
    Indices,
};

struct LocalPivotalMethod1s<'a> {
    tree: Tree<'a>,
    idx: Indices,
    candidates: Vec<usize>,
    history: Vec<usize>,
}

pub fn local_pivotal_method_1s(
    rand: &RandomGenerator,
    probabilities: &[f64],
    eps: f64,
    data: &Matrix,
    bucket_size: usize,
) -> Vec<usize> {
    if probabilities.len() != data.nrow() {
        panic!("size of probability vector must match data");
    }

    let mut pm = PivotalMethod::new(
        LocalPivotalMethod1s {
            tree: Tree::new(data, midpoint_slide, bucket_size),

            idx: Indices::new_fill(probabilities.len()),
            candidates: Vec::<usize>::with_capacity(20),
            history: Vec::<usize>::with_capacity(probabilities.len()),
        },
        &rand,
        probabilities,
        eps,
    );

    pm.run_and_return()
}

impl<'a> DrawVariant for LocalPivotalMethod1s<'a> {
    fn len(&self) -> usize {
        self.idx.len()
    }
    fn remove(&mut self, id: usize) -> &mut Self {
        self.idx.remove(id);
        self.tree.remove_unit(id);
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

        while let Some(&id) = self.history.last() {
            if self.idx.includes(id) {
                break;
            }

            self.history.pop();
        }

        if self.history.len() == 0 {
            self.history.push(self.idx.get_random(rand));
        }

        loop {
            let id1 = *self.history.last().expect("history should have units");
            self.tree.find_neighbours_of_id(id1, 1);
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.tree.get_neighbours());

            let mut i: usize = 0;
            let mut len: usize = self.candidates.len();

            while i < len {
                self.tree.find_neighbours_of_id(self.candidates[i], 1);

                if self.tree.get_neighbours().iter().any(|&id| id == id1) {
                    i += 1;
                } else {
                    len -= 1;
                    self.candidates.swap(i, len);
                }
            }

            if len > 0 {
                let id2 = self.candidates[0..len].draw(rand);
                return Some((id1, id2));
            }

            if self.history.len() == self.tree.population_size() {
                self.history.clear();
                self.history.push(id1);
            }

            self.history.push(self.candidates.draw(rand));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAND00: RandomGenerator = || 0.0;
    // const RAND99: RandomGenerator = || 0.999;

    #[test]
    fn lpm1s_draw() {
        let data = Matrix::new(
            &[
                0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
                0.66079779, 0.62911404, 0.06178627, //
                0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185,
                0.9919061, 0.3800352, 0.7774452,
            ],
            10,
        );

        let mut variant = LocalPivotalMethod1s {
            tree: Tree::new(&data, midpoint_slide, 1),
            idx: Indices::new_fill(10),
            candidates: Vec::<usize>::with_capacity(20),
            history: Vec::<usize>::with_capacity(10),
        };

        assert_eq!(variant.draw(&RAND00), Some((0, 1)));
        // assert_eq!(variant.draw(&rand1), Some((9, 4)));
    }
}
