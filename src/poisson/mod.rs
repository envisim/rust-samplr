use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::probability::Probabilities;
use rand::Rng;

mod correlated_poisson;
pub use correlated_poisson::*;

#[inline]
pub(crate) fn sample_internal<R>(rand: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: Rng + ?Sized,
{
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| (rand.gen::<f64>() <= p).then_some(i))
        .collect()
}

pub fn sample<R>(rand: &mut R, probabilities: &[f64]) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    Probabilities::check(probabilities)?;
    Ok(sample_internal(rand, probabilities))
}

pub fn conditional_sample<R>(
    rand: &mut R,
    probabilities: &[f64],
    sample_size: usize,
    max_iterations: u32,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let population_size = probabilities.len();
    Probabilities::check(probabilities)
        .and(InputError::check_sample_size(sample_size, population_size))?;

    let mut iterations: u32 = 0;

    while iterations < max_iterations {
        let s = sample_internal(rand, probabilities);

        if s.len() == sample_size {
            return Ok(s);
        }

        iterations += 1;
    }

    Err(SamplingError::MaxIterations)
}
