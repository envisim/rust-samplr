use envisim_samplr::{SampleOptions, Sampler};
use envisim_utils::error::SamplingError;
use envisim_utils::utils::sum;
use rand::Rng;

#[allow(dead_code)]
pub fn test_wor<R>(
    sampler: Sampler<R>,
    rng: &mut R,
    opts: &SampleOptions,
    probs: &[f64],
    eps: f64,
    iter: u32,
) -> Result<(), SamplingError>
where
    R: Rng + ?Sized,
{
    let mut sel: Vec<u32> = vec![0; probs.len()];

    for _ in 0..iter {
        opts.sample(rng, sampler)?
            .iter()
            .for_each(|&id| sel[id] += 1);
    }

    let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
    let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

    if !d.iter().all(|&x| x.abs() < eps) {
        panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(&probs), sum(&q));
    }

    Ok(())
}

#[allow(dead_code)]
pub fn test_wor2<F>(mut sampler: F, probs: &[f64], eps: f64, iter: u32) -> Result<(), SamplingError>
where
    F: FnMut() -> Result<Vec<usize>, SamplingError>,
{
    let mut sel: Vec<u32> = vec![0; probs.len()];

    for _ in 0..iter {
        sampler()?.iter().for_each(|&id| sel[id] += 1);
    }

    let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
    let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

    if !d.iter().all(|&x| x.abs() < eps) {
        panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(&probs), sum(&q));
    }

    Ok(())
}
