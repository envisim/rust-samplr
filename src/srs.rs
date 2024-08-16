use envisim_utils::error::{InputError, SamplingError};
use rand::Rng;

pub fn sample_wor<R>(
    rand: &mut R,
    sample_size: usize,
    population_size: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    InputError::check_sample_size(sample_size, population_size)?;

    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for i in 0..population_size {
        if rand.gen_range(0..(population_size - i)) < sample_size - sample.len() {
            sample.push(i);
        }
    }

    Ok(sample)
}

pub fn sample_wr<R>(
    rand: &mut R,
    sample_size: usize,
    population_size: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    InputError::check_sample_size(sample_size, population_size)?;

    let mut sample: Vec<usize> = (0..sample_size)
        .map(|_| rand.gen_range(0..population_size))
        .collect();

    sample.sort_unstable();
    Ok(sample)
}
