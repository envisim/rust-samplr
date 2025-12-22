use envisim_samplr::{
    Enabled,
    ProbabilitiesEqual,
    ProbabilitiesUnequal,
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
use envisim_utils::pips::Probabilities;
use envisim_utils::random::*;
use envisim_utils::utils::usize_to_f64;

pub fn matrix_10_2() -> Matrix<'static> { Matrix::new(&DATA_10_2, 10).unwrap() }
pub fn rng() -> SmallRng { SmallRng::seed_from_u64(42) }

#[allow(dead_code)]
pub fn options_equal_10_2() -> SamplingOptions<'static, ProbabilitiesEqual, Enabled, Enabled> {
    let bal = Matrix::new(&BAL_DATA_10_1, 10).unwrap();
    SamplingOptions::new_equal(10, 2)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}
#[allow(dead_code)]
pub fn options_unequal() -> SamplingOptions<'static, ProbabilitiesUnequal, Enabled, Enabled> {
    let bal = Matrix::new(&BAL_DATA_10_1_P, 10).unwrap();
    SamplingOptions::new(&PROB_10_U)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}
#[allow(dead_code)]
pub fn options_unequal_e() -> SamplingOptions<'static, ProbabilitiesUnequal, Enabled, Enabled> {
    let bal = Matrix::new(&BAL_DATA_10_1_P, 10).unwrap();
    SamplingOptions::new(&PROB_10_E)
        .unwrap()
        .set_balancing(bal)
        .unwrap()
        .set_spreading(matrix_10_2())
        .unwrap()
}

#[allow(dead_code)]
pub fn test_wor<F, P, S, B>(
    mut sampler: F,
    options: &SamplingOptions<'_, P, S, B>,
    eps: f64,
    runs: usize,
) where
    F: FnMut() -> Vec<usize>,
    P: Probabilities,
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
        .slice()
        .iter()
        .zip(prob_emp.iter())
        .map(|(p, p_emp)| p - p_emp)
        .collect();

    println!("{:?}, {:?}", sel, sel.iter().sum::<usize>());

    if !diff.iter().all(|&x| x.abs() < eps) {
        let psum = options.probabilities().slice().iter().sum::<f64>();
        let psum_emp = prob_emp.iter().sum::<f64>();
        panic!("{diff:?} >= {eps}\n(sums: {psum} vs. {psum_emp})",);
    }
}
