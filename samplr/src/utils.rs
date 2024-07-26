use envisim_utils::error::SamplingError;
use envisim_utils::indices::Indices;
use envisim_utils::probability::Probabilities;
use envisim_utils::random_generator::RandomGenerator;

pub struct Sample(Vec<usize>);

impl Sample {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Sample(Vec::<usize>::with_capacity(capacity))
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[inline]
    pub fn add(&mut self, idx: usize) {
        self.0.push(idx);
    }

    #[inline]
    pub fn sort(&mut self) -> &mut Self {
        self.0.sort_unstable();
        self
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.to_vec()
    }

    #[inline]
    pub fn get(&self) -> &[usize] {
        &self.0
    }
}

pub struct Container<'a, R>
where
    R: RandomGenerator,
{
    random: &'a mut R,
    probabilities: Probabilities,
    indices: Indices,
    sample: Sample,
}

impl<'a, R> Container<'a, R>
where
    R: RandomGenerator,
{
    pub fn new(rand: &'a mut R, probabilities: &[f64], eps: f64) -> Result<Self, SamplingError> {
        let population_size = probabilities.len();

        let mut container = Container {
            random: rand,
            probabilities: Probabilities::with_values(probabilities)?,
            indices: Indices::with_fill(population_size),
            sample: Sample::new(population_size),
        };

        container.probabilities.eps = eps;

        for i in 0..population_size {
            container.decide_unit(i)?;
        }

        Ok(container)
    }

    #[inline]
    pub fn random(&mut self) -> &mut R {
        self.random
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

    #[inline]
    pub fn indices_mut(&mut self) -> &mut Indices {
        &mut self.indices
    }

    #[inline]
    pub fn indices_random(&mut self) -> Option<&usize> {
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
    pub fn decide_unit(&mut self, idx: usize) -> Result<Option<bool>, SamplingError> {
        if self.probabilities.is_zero(idx) {
            self.indices.remove(idx)?;
            return Ok(Some(false));
        } else if self.probabilities.is_one(idx) {
            self.indices.remove(idx)?;
            self.sample.add(idx);
            return Ok(Some(true));
        }

        Ok(None)
    }

    #[inline]
    pub fn update_last_unit(&mut self) -> Option<usize> {
        let id = *self.indices.last()?;

        self.probabilities[id] = if self.random.rf64() < self.probabilities[id] {
            1.0
        } else {
            0.0
        };

        Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{data_10_2, gen_rand00, EPS};

    #[test]
    fn decide_unit() {
        let mut rand00 = gen_rand00();
        let (_data, prob) = data_10_2();
        let mut c = Container::new(&mut rand00, &prob, EPS).unwrap();
        c.probabilities_mut()[0] = 1.0;
        c.probabilities_mut()[1] = 0.0;
        assert_eq!(c.decide_unit(0).unwrap(), Some(true));
        assert_eq!(c.decide_unit(1).unwrap(), Some(false));
        assert_eq!(c.decide_unit(2).unwrap(), None);
    }
}
