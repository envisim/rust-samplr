// Copyright (C) 2025 Wilmer Prentius, Anton Grafström.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Pivotal method designs

use envisim_utils::kd_tree::Searcher;
use envisim_utils::pips::Probabilities;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    Enabled,
    SamplingOptions,
};
use envisim_utils::utils::{
    sum,
    usize_to_f64,
};
use rustc_hash::FxHashSet;

pub use crate::SamplingError;
use crate::sample_controller::{
    BasicSampleController,
    SampleController,
    SpreadingSampleController,
};

type Pair = (usize, usize);

trait PivotalMethod<C, R>
where
    C: SampleController,
    R: RandomNumberGenerator,
{
    fn controller(&self) -> &C;
    fn controller_mut(&mut self) -> &mut C;
    fn sample(&mut self, rng: &mut R) -> Vec<usize> {
        self.run(rng);
        self.controller_mut().sample_mut().sort_to_vec()
    }
    fn run(&mut self, rng: &mut R) {
        while let Some(units) = self.select_units(rng) {
            self.update_probabilities(rng, units);
        }

        self.controller_mut()
            .unit_decide_last(rng)
            .expect("last unit to be decided");
    }
    fn select_units(&mut self, rng: &mut R) -> Option<(usize, usize)>;
    fn update_probabilities(&mut self, rng: &mut R, (id1, id2): Pair) {
        let one = self.controller().probabilities().one();

        let p1 = self.controller().probabilities().data()[id1];
        let p2 = self.controller().probabilities().data()[id2];
        let psum = p1 + p2;

        if psum == one {
            if self.controller().draw(rng, one) < p1 {
                self.controller_mut()
                    .unit_set_one(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_zero(id2)
                    .expect("id2 to update");
            } else {
                self.controller_mut()
                    .unit_set_zero(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_one(id2)
                    .expect("id2 to update");
            }
            return;
        }

        if one < psum {
            if self.controller().draw(rng, one + one - psum) < one - p2 {
                self.controller_mut()
                    .unit_set_one(id1)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_and_decide(id2, psum - one)
                    .expect("id2 to update");
            } else {
                self.controller_mut()
                    .unit_set_and_decide(id1, psum - one)
                    .expect("id1 to update");
                self.controller_mut()
                    .unit_set_one(id2)
                    .expect("id2 to update");
            }
            return;
        }

        // psum < one
        if self.controller().probabilities().draw(rng, psum) < p1 {
            self.controller_mut()
                .unit_set_and_decide(id1, psum)
                .expect("id1 to update");
            self.controller_mut()
                .unit_set_zero(id2)
                .expect("id2 to update");
        } else {
            self.controller_mut()
                .unit_set_zero(id1)
                .expect("id1 to update");
            self.controller_mut()
                .unit_set_and_decide(id2, psum)
                .expect("id2 to update");
        }
    }
}

struct SequentialPivotalMethod<P>
where
    P: Probabilities,
{
    controller: BasicSampleController<P>,
    pair: Pair,
}
impl<P, R> PivotalMethod<BasicSampleController<P>, R> for SequentialPivotalMethod<P>
where
    P: Probabilities,
    R: RandomNumberGenerator,
{
    fn controller(&self) -> &BasicSampleController<P> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { &mut self.controller }
    fn select_units(&mut self, _: &mut R) -> Option<(usize, usize)> {
        if self.controller.indices().len() <= 1 {
            return None;
        }

        if !self.controller.indices().contains(self.pair.0) {
            self.pair.0 = self.pair.1;

            while !self.controller.indices().contains(self.pair.0) {
                self.pair.0 += 1;

                if self.pair.0 >= self.controller.population_size() {
                    panic!("spm looped past last unit");
                }
            }

            self.pair.1 = self.pair.0 + 1;
        }

        while !self.controller.indices().contains(self.pair.1) {
            self.pair.1 += 1;

            if self.pair.1 >= self.controller.population_size() {
                panic!("spm looped past last unit");
            }
        }

        Some(self.pair)
    }
}
/// Draw a sample using the sequential pivotal method.
/// A variant of the pivotal method where unit competes in order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = spm(&mut rng, p.try_into()?);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
pub fn spm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
{
    SequentialPivotalMethod {
        controller: options.into(),
        pair: (0, 1),
    }
    .sample(rng)
}

struct RandomPivotalMethod<P>
where
    P: Probabilities,
{
    controller: BasicSampleController<P>,
}
impl<P, R> PivotalMethod<BasicSampleController<P>, R> for RandomPivotalMethod<P>
where
    P: Probabilities,
    R: RandomNumberGenerator,
{
    fn controller(&self) -> &BasicSampleController<P> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { &mut self.controller }
    fn select_units(&mut self, rng: &mut R) -> Option<(usize, usize)> {
        let len = self.controller.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((
                self.controller.indices().list()[0],
                self.controller.indices().list()[1],
            ));
        }

        let id1 = *self.controller.indices().draw(rng).unwrap();
        let k = rng.rusize_to(len - 1);
        let mut id2 = *self.controller.indices().get(k).unwrap();

        if id1 == id2 {
            id2 = *self.controller.indices().last().unwrap();
        }

        Some((id1, id2))
    }
}
/// Draw a sample using the random pivotal method.
/// A variant of the pivotal method where unit competes in a random order.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::random::*;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let s = rpm(&mut rng, p.try_into()?);
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Deville, J. C., & Tille, Y. (1998).
/// Unequal probability sampling without replacement through a splitting method.
/// Biometrika, 85(1), 89-101.
/// <https://doi.org/10.1093/biomet/85.1.89>
// pub fn rpm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
// where
//     R: RandomNumberGenerator,
//     P: Probabilities,
// {
//     let controller: BasicSampleController<P> = options.into();
//     RandomPivotalMethod { controller }.sample(rng)
// }
pub fn rpm<R, P, S, B>(rng: &mut R, options: &SamplingOptions<'_, P, S, B>) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities,
    for<'a> BasicSampleController<P>: From<&'a SamplingOptions<'a, P, S, B>>,
{
    let controller: BasicSampleController<P> = options.into();
    RandomPivotalMethod { controller }.sample(rng)
}

struct LocalPivotalMethod1<'a, P>
where
    P: Probabilities,
{
    controller: SpreadingSampleController<'a, P>,
    candidates: Vec<usize>,
    searcher: Searcher,
}
impl<'a, P, R> PivotalMethod<SpreadingSampleController<'a, P>, R> for LocalPivotalMethod1<'a, P>
where
    P: Probabilities,
    R: RandomNumberGenerator,
{
    fn controller(&self) -> &SpreadingSampleController<'a, P> { &self.controller }
    fn controller_mut(&mut self) -> &mut SpreadingSampleController<'a, P> { &mut self.controller }
    fn select_units(&mut self, rng: &mut R) -> Option<(usize, usize)> {
        let len = self.controller.indices().len();
        if len <= 1 {
            return None;
        } else if len == 2 {
            return Some((
                self.controller.indices().list()[0],
                self.controller.indices().list()[1],
            ));
        }

        loop {
            let id1 = *self.controller.indices().draw(rng).unwrap();
            self.searcher
                .find_neighbours_of_id(self.controller.tree(), id1)
                .unwrap();
            self.candidates.clear();

            // Store potential matches in candidates ... needs to check if any is a match
            self.candidates
                .extend_from_slice(self.searcher.neighbours());

            let mut i = 0usize;

            while i < self.candidates.len() {
                self.searcher
                    .find_neighbours_of_id(self.controller.tree(), self.candidates[i])
                    .unwrap();

                if self.searcher.neighbours().contains(&id1) {
                    i += 1;
                } else {
                    self.candidates.swap_remove(i);
                }
            }

            if !self.candidates.is_empty() {
                let id2 = *rng.relement(&self.candidates).unwrap();
                return Some((id1, id2));
            }
        }
    }
}
/// Draw a sample using the local pivotal method 1.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
///
/// # Examples
/// ```
/// use envisim_samplr::pivotal_method::*;
/// use envisim_utils::{Matrix, random::*};
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let s = SampleOptions::new(&p)?.set_spreading(&m)?.sample(&mut rng, lpm_1)?;
///
/// assert_eq!(s.len(), 5);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
pub fn lpm_1<'a, R, P, B>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, P, Enabled, B>,
) -> Vec<usize>
where
    R: RandomNumberGenerator,
    P: Probabilities + From<&'a SamplingOptions<'a, P, Enabled, B>>,
{
    let controller: SpreadingSampleController<'_, P> = options.into();
    let searcher = Searcher::new_1(controller.tree());
    LocalPivotalMethod1 {
        controller,
        candidates: Vec::<usize>::with_capacity(20),
        searcher,
    }
    .sample(rng)
}
