use envisim_samplr_utils::{probability::Probabilities, random_generator::RandomGenerator};

mod correlated_poisson;
pub use correlated_poisson::*;

pub fn sample<R>(rand: &mut R, probabilities: &[f64]) -> Vec<usize>
where
    R: RandomGenerator,
{
    assert!(Probabilities::check(probabilities));

    probabilities
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if rand.rf64() <= p { Some(i) } else { None })
        .collect()
}

pub fn conditional_sample<R>(
    rand: &mut R,
    probabilities: &[f64],
    sample_size: usize,
    max_iterations: u32,
) -> Result<Vec<usize>, &'static str>
where
    R: RandomGenerator,
{
    let population_size = probabilities.len();
    assert!(Probabilities::check(probabilities));
    assert!(sample_size < population_size);

    let mut iterations: u32 = 0;

    while iterations < max_iterations {
        let s = sample(rand, probabilities);

        if s.len() == sample_size {
            return Ok(s);
        }

        iterations += 1;
    }

    Err("could not find a sample")
}
