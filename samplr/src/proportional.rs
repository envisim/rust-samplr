use crate::poisson::sample_internal;
use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::indices::Indices;
use envisim_utils::probability::Probabilities;
use envisim_utils::utils::{sum, usize_to_f64};
use rand::Rng;

/// Assumes probabilites sum to 1.0
#[inline]
fn draw<R>(rand: &mut R, probabilities: &[f64]) -> usize
where
    R: Rng + ?Sized,
{
    let population_size = probabilities.len();
    let rv = rand.gen::<f64>();
    let mut psum: f64 = 0.0;

    for (i, &p) in probabilities.iter().enumerate() {
        psum += p;

        if rv <= psum {
            return i;
        }
    }

    population_size - 1
}

pub fn draw_probabilities_sample<R>(
    rand: &mut R,
    probabilities: &[f64],
    eps: f64,
    n: usize,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    Probabilities::check(probabilities)?;
    if !(1.0 - eps..1.0 + eps).contains(&sum(probabilities)) {
        return Err(SamplingError::Input(InputError::NotInteger));
    }

    if n == 0 {
        return Ok(vec![]);
    }

    let mut rvs = Vec::<f64>::with_capacity(n);

    for _ in 0..n {
        rvs.push(rand.gen::<f64>());
    }

    rvs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sample = Vec::<usize>::with_capacity(n);
    let mut psum: f64 = 0.0;
    let mut rv_iter = rvs.iter();
    let mut rv = *rv_iter.next().unwrap();

    // Add units for which rv is in [psum, psum+p)
    // Go up one p when psum+p < rv
    // Go up one rv when sample has been pushed
    'outer: for (id, &p) in probabilities.iter().enumerate() {
        loop {
            if psum + p <= rv {
                psum += p;
                break;
            }

            if rv < psum + p {
                sample.push(id);

                match rv_iter.next() {
                    Some(v) => {
                        rv = *v;
                        continue;
                    }
                    _ => break 'outer,
                }
            }
        }
    }

    Ok(sample)
}

pub fn sampford<R>(
    rand: &mut R,
    probabilities: &[f64],
    eps: f64,
    max_iterations: u32,
) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;
    let sample_size = psum.round() as usize;

    if sample_size == 0 {
        return Ok(vec![]);
    } else if sample_size == 1 {
        return Ok(vec![draw(rand, probabilities)]);
    }

    let norm_probs: Vec<f64> = probabilities.iter().map(|&p| p / psum).collect();

    for _ in 0..max_iterations {
        let mut sample = sample_internal(rand, probabilities);

        if sample.len() != sample_size - 1 {
            continue;
        }

        let a_unit = draw(rand, &norm_probs);

        // Since sample is ordered, we don't need to check units with
        // higher id than a_unit
        if sample
            .iter()
            .find(|&&id| id >= a_unit)
            .is_some_and(|&id| id != a_unit)
        {
            sample.push(a_unit);
            sample.sort_unstable();
            return Ok(sample);
        }
    }

    Err(SamplingError::MaxIterations)
}

/// Rosén, B. (2000).
/// A user’s guide to Pareto pi-ps sampling. R & D Report 2000:6.
/// Stockholm: Statistiska Centralbyrån.
pub fn pareto<R>(rand: &mut R, probabilities: &[f64], eps: f64) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;

    let sample_size = psum.round() as usize;

    let q_values: Vec<f64> = probabilities
        .iter()
        .map(|&p| {
            let u = rand.gen::<f64>();

            if 1.0 - eps < u || p < eps {
                return f64::INFINITY;
            }

            let res = (u * (1.0 - p)) / (p * (1.0 - u));

            if res.is_nan() {
                return f64::INFINITY;
            }

            res
        })
        .collect();

    let mut sample: Vec<usize> = (0..probabilities.len()).collect();
    sample.sort_by(|&a, &b| q_values[a].partial_cmp(&q_values[b]).unwrap());
    sample.truncate(sample_size);
    Ok(sample)
}

pub fn brewer<R>(rand: &mut R, probabilities: &[f64], eps: f64) -> Result<Vec<usize>, SamplingError>
where
    R: Rng + ?Sized,
{
    let mut psum = sum(probabilities);
    Probabilities::check(probabilities)
        .and(Probabilities::check_eps(eps))
        .and(InputError::check_integer_approx(psum, eps))?;

    let mut sample_size = psum.round() as usize;
    let mut n_d = psum;
    let mut indices = Indices::with_fill(probabilities.len());
    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for (id, &p) in probabilities.iter().enumerate() {
        if p <= eps {
            indices.remove(id).unwrap();
        } else if 1.0 - eps <= p {
            indices.remove(id).unwrap();
            sample.push(id);
            n_d -= 1.0;
            sample_size -= 1;
        }
    }

    let mut q_probs: Vec<f64> = vec![0.0; probabilities.len()];

    for i in 0..sample_size {
        psum = 0.0;
        for &id in indices.list() {
            let p = probabilities[id];
            q_probs[id] = p * (n_d - p) / (n_d - p * usize_to_f64(sample_size - i + 1));
            psum += q_probs[id];
        }

        for &id in indices.list() {
            q_probs[id] /= psum;
        }

        let a_unit = draw(rand, &q_probs);
        indices.remove(a_unit).unwrap();
        sample.push(a_unit);
        q_probs[a_unit] = 0.0;
        n_d -= probabilities[a_unit];
    }

    sample.sort_unstable();
    Ok(sample)
}
