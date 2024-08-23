use envisim_samplr::cube_method::*;
use envisim_test_utils::*;
use envisim_utils::utils::sum;
use envisim_utils::Matrix;

mod test_utils;
use test_utils::*;

const BAL_DATA_10_1: [f64; 10] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
];
const BAL_DATA_10_1_P: [f64; 20] = [
    0.20f64, 0.25, 0.35, 0.40, 0.50, 0.50, 0.55, 0.65, 0.70, 0.90, //
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
];

#[test]
fn test_cube() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let baldata = Matrix::from_ref(&BAL_DATA_10_1_P, 10);
    let mut opts = SampleOptions::new(p)?;
    opts.balancing(&baldata)?;

    test_wor(cube, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_lcube() -> Result<(), SamplingError> {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = Matrix::from_ref(&DATA_10_2, 10);
    let baldata = Matrix::from_ref(&BAL_DATA_10_1_P, 10);
    let mut opts = SampleOptions::new(p)?;
    opts.balancing(&baldata)?.auxiliaries(&data)?;

    test_wor(local_cube, &mut rng, &opts, p, 1e-2, 100000)
}

#[test]
fn test_cube_stratified() -> Result<(), SamplingError> {
    let eps = 1e-2;
    let iter = 100000;

    let mut rng = seeded_rng();
    let probs = &PROB_10_E;
    let baldata = Matrix::from_ref(&BAL_DATA_10_1, 10);
    let mut opts = SampleOptions::new(probs)?;
    opts.balancing(&baldata)?;

    {
        let mut sel: Vec<u32> = vec![0; probs.len()];

        for _ in 0..iter {
            let s = cube_stratified(&mut rng, &opts, &[1i64, 1, 1, 1, 1, 2, 2, 2, 2, 2])?;
            assert_eq!(s.len(), 2);
            assert!((0..5).contains(&s[0]));
            assert!((5..10).contains(&s[1]));
            s.iter().for_each(|&id| sel[id] += 1);
        }

        let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
        let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

        if !d.iter().all(|&x| x.abs() < eps) {
            panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(probs), sum(&q));
        }
    }
    Ok(())
}

#[test]
fn test_lcube_stratified() -> Result<(), SamplingError> {
    let eps = 1e-2;
    let iter = 100000;

    let mut rng = seeded_rng();
    let probs = &PROB_10_U;
    let data = Matrix::from_ref(&DATA_10_2, 10);
    let baldata = Matrix::from_ref(&BAL_DATA_10_1, 10);
    let mut opts = SampleOptions::new(probs)?;
    opts.balancing(&baldata)?.auxiliaries(&data)?;

    {
        let mut sel: Vec<u32> = vec![0; probs.len()];

        for _ in 0..iter {
            let s = local_cube_stratified(&mut rng, &opts, &[1i64, 1, 1, 1, 1, 2, 2, 2, 3, 3])?;
            s.iter().for_each(|&id| sel[id] += 1);
        }

        let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
        let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

        if !d.iter().all(|&x| x.abs() < eps) {
            panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(probs), sum(&q));
        }
    }
    Ok(())
}
