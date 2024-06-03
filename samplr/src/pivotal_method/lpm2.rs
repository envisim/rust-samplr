use crate::pivotal_method::base::{DrawVariant, PivotalMethod};
use envisim_samplr_utils::{
    kd_tree::{midpoint_slide, Tree},
    matrix::Matrix,
    uniform_random::{DrawRandom, RandomGenerator},
    Indices,
};

struct LocalPivotalMethod2<'a> {
    tree: Tree<'a>,
    idx: Indices,
}

pub fn local_pivotal_method_2(
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
        LocalPivotalMethod2 {
            tree: Tree::new(data, midpoint_slide, bucket_size),
            idx: Indices::new_fill(probabilities.len()),
        },
        &rand,
        probabilities,
        eps,
    );

    pm.run_and_return()
}

impl<'a> DrawVariant for LocalPivotalMethod2<'a> {
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

        let id1 = self.idx.get_random(rand);
        self.tree.find_neighbours_of_id(id1, 1);

        let id2 = self.tree.get_neighbours().draw(rand);

        Some((id1, id2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAND00: RandomGenerator = || 0.0;
    // const RAND99: RandomGenerator = || 0.999;

    #[test]
    fn lpm2() {
        let data = Matrix::new(
            &[
                0.26550866, 0.37212390, 0.57285336, 0.90820779, 0.20168193, 0.89838968, 0.94467527,
                0.66079779, 0.62911404, 0.06178627, //
                0.2059746, 0.1765568, 0.6870228, 0.3841037, 0.7698414, 0.4976992, 0.7176185,
                0.9919061, 0.3800352, 0.7774452,
            ],
            10,
        );

        let lpm = local_pivotal_method_2(
            &RAND00,
            &vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            1e-12,
            &data,
            1,
        );

        let result: Vec<usize> = vec![1, 3];
        assert_eq!(lpm, result);
    }
}
