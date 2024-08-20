use envisim_utils::error::SamplingError;
use envisim_utils::utils::sum;

pub fn test_wor<F>(mut sampler: F, probs: &[f64], eps: f64, iter: u32)
where
    F: FnMut() -> Result<Vec<usize>, SamplingError>,
{
    let mut sel: Vec<u32> = vec![0; probs.len()];

    for _ in 0..iter {
        sampler().unwrap().iter().for_each(|&id| sel[id] += 1);
    }

    let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
    let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

    if !d.iter().all(|&x| x.abs() < eps) {
        panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(&probs), sum(&q));
    }
}
