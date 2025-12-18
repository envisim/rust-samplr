// Copyright (C) 2025 Wilmer Prentius.
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

use envisim_utils::indices::Indices;
use envisim_utils::kd_tree::{
    Node,
    TreeBuilder,
};
use envisim_utils::probabilities::{
    Probabilities,
    ProbabilitiesEqual,
    ProbabilitiesUnequal,
};
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sampling_options::{
    Enabled,
    SamplingOptions,
    SpreadingOptions,
};

pub struct Sample(Vec<usize>);
impl Sample {
    pub fn new(capacity: usize) -> Self { Sample(Vec::<usize>::with_capacity(capacity)) }
    pub fn clear(&mut self) { self.0.clear(); }
    pub fn add(&mut self, idx: usize) { self.0.push(idx); }
    pub fn sort(&mut self) -> &mut Self {
        self.0.sort_unstable();
        self
    }
    pub fn to_vec(&self) -> Vec<usize> { self.0.to_vec() }
    pub fn sort_to_vec(&mut self) -> Vec<usize> { self.sort().to_vec() }
    pub fn get(&self) -> &[usize] { &self.0 }
    pub fn len(&self) -> usize { self.0.len() }
}

pub enum DecideUnit {
    In,
    Out,
    Undecided,
}

pub trait SampleController {
    type Probs: Probabilities;
    fn controller(&self) -> &BasicSampleController<Self::Probs>;
    fn controller_mut(&mut self) -> &mut BasicSampleController<Self::Probs>;
    fn probabilities(&self) -> &Self::Probs { &self.controller().probabilities }
    fn probabilities_mut(&mut self) -> &mut Self::Probs { &mut self.controller_mut().probabilities }
    fn indices(&self) -> &Indices { &self.controller().indices }
    fn indices_mut(&mut self) -> &mut Indices { &mut self.controller_mut().indices }
    fn population_size(&self) -> usize { self.probabilities().len() }

    fn sample(&self) -> &Sample { &self.controller().sample }
    fn sample_mut(&mut self) -> &mut Sample { &mut self.controller_mut().sample }

    fn draw<R>(
        &self,
        rng: &mut R,
        max: <<Self as SampleController>::Probs as Probabilities>::Prob,
    ) -> <<Self as SampleController>::Probs as Probabilities>::Prob
    where
        R: RandomNumberGenerator,
    {
        self.probabilities().draw(rng, max)
    }

    fn unit_remove(&mut self, idx: usize) -> Option<usize> { self.indices_mut().remove(idx) }
    fn unit_decide(&mut self, idx: usize) -> Option<DecideUnit> {
        if self.probabilities().is_one(idx) {
            self.sample_mut().add(idx);
            self.unit_remove(idx)?;
            return Some(DecideUnit::In);
        } else if self.probabilities().is_zero(idx) {
            self.unit_remove(idx)?;
            return Some(DecideUnit::Out);
        }
        Some(DecideUnit::Undecided)
    }
    fn unit_set_and_decide(
        &mut self,
        idx: usize,
        prob: <<Self as SampleController>::Probs as Probabilities>::Prob,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().set(idx, prob).unwrap();
        self.unit_decide(idx)
    }
    fn unit_set_one(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_one(idx);
        self.sample_mut().add(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::In)
    }
    fn unit_set_zero(&mut self, idx: usize) -> Option<DecideUnit> {
        self.probabilities_mut().set_zero(idx);
        self.unit_remove(idx)?;
        Some(DecideUnit::Out)
    }
    fn unit_add_and_decide(
        &mut self,
        idx: usize,
        prob: <<Self as SampleController>::Probs as Probabilities>::Prob,
    ) -> Option<DecideUnit> {
        self.probabilities_mut().add(idx, prob);
        self.unit_decide(idx)
    }
    fn unit_decide_last<R>(&mut self, rng: &mut R) -> Option<DecideUnit>
    where
        R: RandomNumberGenerator,
    {
        let Some(id) = self.indices().last() else {
            return Some(DecideUnit::Undecided);
        };

        if self.probabilities().data()[id] < self.draw(rng, self.probabilities().one()) {
            self.unit_set_one(id)
        } else {
            self.unit_set_zero(id)
        }
    }
}

pub struct BasicSampleController<P = ProbabilitiesUnequal>
where
    P: Probabilities,
{
    probabilities: P,
    indices: Indices,
    sample: Sample,
}
pub struct SpreadingSampleController<'a, P = ProbabilitiesUnequal>
where
    P: Probabilities,
{
    controller: BasicSampleController<P>,
    tree: Box<Node<'a>>,
}

impl<P> BasicSampleController<P>
where
    P: Probabilities,
{
    fn init(mut self) -> Self {
        let population_size = self.population_size();
        for i in 0..population_size {
            self.unit_decide(i).unwrap();
        }

        self
    }
}

impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesUnequal, S, B>>
    for BasicSampleController<ProbabilitiesUnequal>
{
    fn from(
        options: &SamplingOptions<'a, ProbabilitiesUnequal, S, B>,
    ) -> BasicSampleController<ProbabilitiesUnequal> {
        let population_size = options.population_size();
        let controller = BasicSampleController::<ProbabilitiesUnequal> {
            probabilities: options.into(),
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };
        controller.init()
    }
}
impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesEqual, S, B>>
    for BasicSampleController<ProbabilitiesUnequal>
{
    fn from(
        options: &SamplingOptions<'a, ProbabilitiesEqual, S, B>,
    ) -> BasicSampleController<ProbabilitiesUnequal> {
        let population_size = options.population_size();
        let controller = BasicSampleController::<ProbabilitiesUnequal> {
            probabilities: options.into(),
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };
        controller.init()
    }
}
impl<'a, S, B> From<&SamplingOptions<'a, ProbabilitiesEqual, S, B>>
    for BasicSampleController<ProbabilitiesEqual>
{
    fn from(
        options: &SamplingOptions<'a, ProbabilitiesEqual, S, B>,
    ) -> BasicSampleController<ProbabilitiesEqual> {
        let population_size = options.population_size();
        let controller = BasicSampleController::<ProbabilitiesEqual> {
            probabilities: options.into(),
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };
        controller.init()
    }
}
impl<'a, B> From<&'a SamplingOptions<'a, ProbabilitiesUnequal, Enabled, B>>
    for SpreadingSampleController<'a, ProbabilitiesUnequal>
{
    fn from(
        options: &'a SamplingOptions<'a, ProbabilitiesUnequal, Enabled, B>,
    ) -> SpreadingSampleController<'a, ProbabilitiesUnequal> {
        let controller: BasicSampleController<ProbabilitiesUnequal> = options.into();
        let mut units = controller.indices().to_vec();

        SpreadingSampleController::<'a, ProbabilitiesUnequal> {
            controller,
            tree: options.spreading().build(&mut units).unwrap().into(),
        }
    }
}
impl<'a, B> From<&'a SamplingOptions<'a, ProbabilitiesEqual, Enabled, B>>
    for SpreadingSampleController<'a, ProbabilitiesUnequal>
{
    fn from(
        options: &'a SamplingOptions<'a, ProbabilitiesEqual, Enabled, B>,
    ) -> SpreadingSampleController<'a, ProbabilitiesUnequal> {
        let controller: BasicSampleController<ProbabilitiesUnequal> = options.into();
        let mut units = controller.indices().to_vec();

        SpreadingSampleController::<'a, ProbabilitiesUnequal> {
            controller,
            tree: options.spreading().build(&mut units).unwrap().into(),
        }
    }
}
impl<'a, B> From<&'a SamplingOptions<'a, ProbabilitiesEqual, Enabled, B>>
    for SpreadingSampleController<'a, ProbabilitiesEqual>
{
    fn from(
        options: &'a SamplingOptions<'a, ProbabilitiesEqual, Enabled, B>,
    ) -> SpreadingSampleController<'a, ProbabilitiesEqual> {
        let controller: BasicSampleController<ProbabilitiesEqual> = options.into();
        let mut units = controller.indices().to_vec();

        SpreadingSampleController::<'a, ProbabilitiesEqual> {
            controller,
            tree: options.spreading().build(&mut units).unwrap().into(),
        }
    }
}

impl<P> SampleController for BasicSampleController<P>
where
    P: Probabilities,
{
    type Probs = P;
    fn controller(&self) -> &BasicSampleController<P> { self }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { self }
}
impl<'a, P> SampleController for SpreadingSampleController<'a, P>
where
    P: Probabilities,
{
    type Probs = P;
    fn controller(&self) -> &BasicSampleController<P> { &self.controller }
    fn controller_mut(&mut self) -> &mut BasicSampleController<P> { &mut self.controller }
    fn unit_remove(&mut self, idx: usize) -> Option<usize> {
        self.tree.remove_unit(idx)?;
        self.controller.unit_remove(idx)
    }
}

impl<'a, P> SpreadingSampleController<'a, P>
where
    P: Probabilities,
{
    pub fn tree(&self) -> &Node<'a> { &self.tree }
    pub fn tree_mut(&mut self) -> &mut Node<'a> { &mut self.tree }
    pub fn reset_tree(
        &mut self,
        options: &'a SpreadingOptions<'a>,
        units: &mut [usize],
    ) -> Option<()> {
        self.tree = options.build(units)?.into();
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use envisim_test_utils::*;

    use super::*;

    #[test]
    fn equal_unequal() {
        let oe = SamplingOptions::new_equal(10, 2).unwrap();
        let ou = SamplingOptions::new(&PROB_10_E).unwrap();
        assert_eq!(
            oe.probabilities().slice().as_ref(),
            ou.probabilities().slice().as_ref(),
        );

        let ce: BasicSampleController<ProbabilitiesUnequal> = (&oe).into();
        let cu: BasicSampleController<ProbabilitiesUnequal> = (&ou).into();
        assert_eq!(ce.probabilities().data(), cu.probabilities().data());
    }
}
