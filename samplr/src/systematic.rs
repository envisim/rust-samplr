use envisim_samplr_utils::error::SamplingError;
use envisim_samplr_utils::probability::Probabilities;
use envisim_samplr_utils::random_generator::RandomGenerator;

pub fn sample<R>(rand: &mut R, probabilities: &[f64]) -> Result<Vec<usize>, SamplingError>
where
    R: RandomGenerator,
{
    let order: Vec<usize> = (0usize..probabilities.len()).collect();
    from_order(rand, probabilities, &order)
}

pub fn sample_random_order<R>(
    rand: &mut R,
    probabilities: &[f64],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomGenerator,
{
    let order = shuffle(rand, probabilities.len());
    from_order(rand, probabilities, &order)
}

fn from_order<R>(
    rand: &mut R,
    probabilities: &[f64],
    order: &[usize],
) -> Result<Vec<usize>, SamplingError>
where
    R: RandomGenerator,
{
    Probabilities::check(probabilities)?;

    let mut sample = Vec::<usize>::with_capacity(
        probabilities.iter().fold(0.0, |acc, p| acc + p).ceil() as usize,
    );
    let mut r = rand.rf64();
    let mut psum: f64 = 0.0;

    for &id in order.iter() {
        if psum <= r && r <= psum + probabilities[id] {
            sample.push(id);
            r += 1.0;
        }

        psum += probabilities[id];
    }

    Ok(sample)
}

fn shuffle<R>(rand: &mut R, len: usize) -> Vec<usize>
where
    R: RandomGenerator,
{
    let mut order: Vec<usize> = (0..len).collect();

    for i in (1..=len).rev() {
        order.swap(i, rand.rusize(i + 1));
    }

    order
}
