use envisim_samplr_utils::error::{InputError, SamplingError};
use envisim_samplr_utils::{probability::Probabilities, random_generator::RandomGenerator};

mod correlated_poisson;
pub use correlated_poisson::*;

#[inline]
pub(crate) fn sample_internal<R>(rand: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: RandomGenerator,
{
    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if rand.rf64() <= p { Some(i) } else { None })
        .collect()
}

pub fn sample<R>(rand: &mut R, probabilities: &[f64]) -> Result<Vec<usize>, SamplingError>
where
    R: RandomGenerator,
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
    R: RandomGenerator,
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
