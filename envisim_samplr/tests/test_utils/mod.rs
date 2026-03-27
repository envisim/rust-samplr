use envisim_samplr::{
    ProbabilitySpec,
    ProbabilitySpecEqual,
    ProbabilitySpecUnequal,
    SamplingOptions,
};
use envisim_test_utils::{
    BAL_DATA_10_1,
    BAL_DATA_10_1_P,
    DATA_10_2,
    PROB_10_E,
    PROB_10_U,
};
use envisim_utils::matrix::Matrix;
use envisim_utils::random::*;
use envisim_utils::utils::usize_to_f64;

pub fn matrix_10_2() -> Matrix<'static> { Matrix::new(&DATA_10_2, 10).unwrap() }
pub fn rng() -> SmallRng { SmallRng::seed_from_u64(42) }

#[allow(dead_code)]
pub fn options_equal_10_2() -> SamplingOptions<'static, ProbabilitySpecEqual> {
    let bal = Matrix::new(&BAL_DATA_10_1, 10).unwrap();
    SamplingOptions::new_equal(10, 2)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}
#[allow(dead_code)]
pub fn options_unequal() -> SamplingOptions<'static, ProbabilitySpecUnequal<'static>> {
    let bal = Matrix::new(&BAL_DATA_10_1_P, 10).unwrap();
    SamplingOptions::new(&PROB_10_U)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}
#[allow(dead_code)]
pub fn options_unequal_e() -> SamplingOptions<'static, ProbabilitySpecUnequal<'static>> {
    let bal = Matrix::new(&BAL_DATA_10_1_P, 10).unwrap();
    SamplingOptions::new(&PROB_10_E)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}
#[allow(dead_code)]
pub fn matrix_big_balanced() -> ([f64; 1000], Vec<f64>) {
    const P_VEC: [f64; 1000] = [0.1; 1000];
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

    (P_VEC, b_vec)
}

#[allow(dead_code)]
pub fn test_wor<F, PS>(mut sampler: F, options: &SamplingOptions<'_, PS>, eps: f64, runs: usize)
where
    F: FnMut() -> Vec<usize>,
    PS: ProbabilitySpec,
{
    let mut sel: Vec<usize> = vec![0; options.population_size()];

    for _ in 0..runs {
        sampler().iter().for_each(|&id| sel[id] += 1);
    }

    let prob_emp: Vec<f64> = sel
        .iter()
        .map(|&s| usize_to_f64(s) / usize_to_f64(runs))
        .collect();
    let diff: Vec<f64> = options
        .probabilities()
        .as_f64_slice()
        .iter()
        .zip(prob_emp.iter())
        .map(|(p, p_emp)| p - p_emp)
        .collect();

    println!("{:?}, {:?}", sel, sel.iter().sum::<usize>());

    if !diff.iter().all(|&x| x.abs() < eps) {
        let psum = options.probabilities().sample_size_f64();
        let psum_emp = prob_emp.iter().sum::<f64>();
        panic!("{diff:?} >= {eps}\n(sums: {psum} vs. {psum_emp})",);
    }
}

#[allow(dead_code)]
pub fn test_wor_fixed_n<F, PS>(mut sampler: F, options: &SamplingOptions<'_, PS>, runs: usize)
where
    F: FnMut() -> Vec<usize>,
    PS: ProbabilitySpec,
{
    let mut sel: Vec<usize> = vec![0; options.population_size()];
    let expected_n = options.probabilities().sample_size();

    for r in 0..runs {
        let s = sampler();
        let s_len = s.len();
        assert_eq!(
            s_len, expected_n,
            "sample size {s_len} should be {expected_n} (run {r})"
        );
        sampler().iter().for_each(|&id| sel[id] += 1);
    }
}
