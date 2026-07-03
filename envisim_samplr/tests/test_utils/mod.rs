use envisim_samplr::{
    ProbabilityOptions,
    RealUnequalProbabilityOptions,
    SamplingOptions,
    UnequalProbabilityOptions,
};
use envisim_utils::matrix::Matrix;
use envisim_utils::random::*;
pub use envisim_utils::test_utils::*;
use num_traits::ToPrimitive;

fn rng() -> SmallRng { SmallRng::seed_from_u64(42) }

#[allow(dead_code)]
pub fn matrix_big_balanced() -> (
    UnequalProbabilityOptions<RealUnequalProbabilityOptions<&'static [f64]>>,
    Matrix<f64>,
) {
    const P_VEC: [f64; 1000] = [0.1; 1000];
    let spec = UnequalProbabilityOptions::new(P_VEC.as_ref()).unwrap();

    let mut b_vec: Vec<f64> = Vec::with_capacity(2000);
    b_vec.extend_from_slice(&P_VEC);
    b_vec.extend_from_slice(&[1.0; 101]);
    b_vec.extend_from_slice(&[0.0; 899]);
    b_vec.extend_from_slice(&[1.0; 201]);
    b_vec.extend_from_slice(&[0.0; 799]);
    b_vec.extend_from_slice(&[1.0; 301]);
    b_vec.extend_from_slice(&[0.0; 699]);
    b_vec.extend_from_slice(&[1.0; 401]);
    b_vec.extend_from_slice(&[0.0; 599]);
    b_vec.extend_from_slice(&[1.0; 501]);
    b_vec.extend_from_slice(&[0.0; 499]);
    b_vec.extend_from_slice(&[1.0; 601]);
    b_vec.extend_from_slice(&[0.0; 399]);
    b_vec.extend_from_slice(&[1.0; 701]);
    b_vec.extend_from_slice(&[0.0; 299]);
    b_vec.extend_from_slice(&[1.0; 801]);
    b_vec.extend_from_slice(&[0.0; 199]);
    b_vec.extend_from_slice(&[1.0; 901]);
    b_vec.extend_from_slice(&[0.0; 99]);
    // P2
    b_vec.extend_from_slice(&[1.0; 100]);
    b_vec.extend_from_slice(&[0.0; 800]);
    b_vec.extend_from_slice(&[1.0; 100]);
    // P3
    b_vec.extend_from_slice(&[1.0; 50]);
    b_vec.extend_from_slice(&[0.0; 800]);
    b_vec.extend_from_slice(&[1.0; 150]);
    // P4
    b_vec.extend_from_slice(&[1.0; 150]);
    b_vec.extend_from_slice(&[0.0; 400]);
    b_vec.extend_from_slice(&[1.0; 450]);

    let mat = Matrix::new(b_vec, nz(1000)).unwrap();

    (spec, mat)
}

#[allow(dead_code)]
#[inline]
pub fn test_wor<F, PO, SOP, BOP>(
    mut sampler: F,
    options: &SamplingOptions<PO, SOP, BOP>,
    eps: f64,
    runs: usize,
) where
    F: FnMut(&mut SmallRng) -> Vec<usize>,
    PO: ProbabilityOptions<Real = f64>,
{
    let mut sel: Vec<usize> = vec![0; options.population_size().get()];
    let sample_size = options.sample_size();

    let mut rng = rng();
    for _ in 0..runs {
        let s = sampler(&mut rng);
        assert_eq!(
            s.len(),
            sample_size,
            "expected sample size of {}, got {}",
            sample_size,
            s.len()
        );
        s.iter().for_each(|&id| sel[id] += 1);
    }

    let prob_emp: Vec<f64> = sel
        .iter()
        .map(|&s| s.to_f64().unwrap() / runs.to_f64().unwrap())
        .collect();
    let diff: Vec<f64> = options
        .probabilities()
        .iter_real()
        .zip(prob_emp.iter())
        .map(|(p, p_emp)| p - p_emp)
        .collect();

    println!("{:?}, {:?}", sel, sel.iter().sum::<usize>());

    if !diff.iter().all(|&x| x.abs() < eps) {
        let psum = options.probabilities().sample_size_real();
        let psum_emp = prob_emp.iter().sum::<f64>();
        panic!("{diff:?} >= {eps}\n(sums: {psum} vs. {psum_emp})",);
    }
}

#[allow(dead_code)]
#[inline]
pub fn test_wor_random_n<F, PO, SOP, BOP>(
    mut sampler: F,
    options: &SamplingOptions<PO, SOP, BOP>,
    eps: f64,
    runs: usize,
) where
    F: FnMut(&mut SmallRng) -> Vec<usize>,
    PO: ProbabilityOptions<Real = f64>,
{
    let mut sel: Vec<usize> = vec![0; options.population_size().get()];

    let mut rng = rng();
    for _ in 0..runs {
        let s = sampler(&mut rng);
        s.iter().for_each(|&id| sel[id] += 1);
    }

    let prob_emp: Vec<f64> = sel
        .iter()
        .map(|&s| s.to_f64().unwrap() / runs.to_f64().unwrap())
        .collect();
    let diff: Vec<f64> = options
        .probabilities()
        .iter_real()
        .zip(prob_emp.iter())
        .map(|(p, p_emp)| p - p_emp)
        .collect();

    println!("{:?}, {:?}", sel, sel.iter().sum::<usize>());

    if !diff.iter().all(|&x| x.abs() < eps) {
        let psum = options.probabilities().sample_size_real();
        let psum_emp = prob_emp.iter().sum::<f64>();
        panic!("{diff:?} >= {eps}\n(sums: {psum} vs. {psum_emp})",);
    }
}
