use crate::kd_tree::{Node, Searcher};
use crate::matrix::{OperateMatrix, RefMatrix};
use crate::probability::Probabilities;
use crate::utils::{sum, usize_to_f64};

#[inline]
pub fn estimate(y_values: &[f64], probabilities: &[f64]) -> f64 {
    assert_eq!(y_values.len(), probabilities.len());
    assert!(Probabilities::check(probabilities));

    y_values
        .iter()
        .zip(probabilities.iter())
        .fold(0.0, |acc, (&y, &p)| acc + y / p)
}

#[inline]
pub fn ratio(y_values: &[f64], x_values: &[f64], probabilities: &[f64], x_total: f64) -> f64 {
    assert!(x_total > 0.0);
    estimate(y_values, probabilities) / estimate(x_values, probabilities) * x_total
}

pub fn variance(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &RefMatrix,
) -> f64 {
    let sample_size = y_values.len();
    assert_eq!(sample_size, probabilities.len());
    assert_eq!(sample_size, probabilities_second_order.nrow());
    assert_eq!(sample_size, probabilities_second_order.ncol());
    assert!(Probabilities::check(probabilities));
    assert!(Probabilities::check(probabilities_second_order.data()));

    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let y_pi = y_values[i] / probabilities[i];
        variance += y_pi.powi(2) * (1.0 - probabilities[i]);

        for j in (i + 1)..sample_size {
            variance += 2.0 * y_pi * y_values[j] / probabilities[j]
                * (1.0 - probabilities[i] * probabilities[j] / probabilities_second_order[(i, j)]);
        }
    }

    variance
}

pub fn syg_variance(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &RefMatrix,
) -> f64 {
    let sample_size = y_values.len();
    assert_eq!(sample_size, probabilities.len());
    assert_eq!(sample_size, probabilities_second_order.nrow());
    assert_eq!(sample_size, probabilities_second_order.ncol());
    assert!(Probabilities::check(probabilities));
    assert!(Probabilities::check(probabilities_second_order.data()));

    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let y_pi = y_values[i] / probabilities[i];

        for j in (i + 1)..sample_size {
            variance -= (y_pi - y_values[j] / probabilities[j]).powi(2)
                * (1.0 - probabilities[i] * probabilities[j] / probabilities_second_order[(i, j)]);
        }
    }

    variance
}

pub fn deville_variance(y_values: &[f64], probabilities: &[f64]) -> f64 {
    let y_pi: Vec<f64> = y_values
        .iter()
        .zip(probabilities.iter())
        .map(|(&y, &p)| y / p)
        .collect();

    let q: Vec<f64> = probabilities.iter().map(|&p| 1.0 - p).collect();

    let s1mp = sum(&q);
    let del = y_pi
        .iter()
        .zip(q.iter())
        .fold(0.0, |acc, (&a, &b)| acc + a * b);
    let s1mp_del = s1mp / del;
    let sak2 = q.iter().fold(0.0, |acc, &a| acc + a.powi(2)) / s1mp.powi(2);

    let dsum = y_pi
        .iter()
        .zip(q.iter())
        .fold(0.0, |acc, (&a, &b)| acc + (a - s1mp_del).powi(2) * b);

    1.0 / (1.0 - sak2) * dsum
}

pub fn local_mean_variance(
    y_values: &[f64],
    probabilities: &[f64],
    auxilliaries: &RefMatrix,
    n_neighbours: usize,
    bucket_size: usize,
) -> f64 {
    let sample_size = y_values.len();
    assert_eq!(sample_size, probabilities.len());
    assert!(Probabilities::check(probabilities));
    assert!(n_neighbours > 1);

    let tree = {
        let mut units: Vec<usize> = (0..sample_size).collect();
        Node::new_midpoint_slide(bucket_size, auxilliaries, &mut units)
    };
    let mut searcher = Searcher::new(&tree, n_neighbours);

    let yp: Vec<f64> = y_values
        .iter()
        .zip(probabilities.iter())
        .map(|(&y, &p)| y / p)
        .collect();
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        searcher.find_neighbours_of_iter(&tree, &mut auxilliaries.into_row_iter(i));
        let len = usize_to_f64(searcher.neighbours().len());
        variance += len / (len - 1.0)
            * (searcher
                .neighbours()
                .iter()
                .fold(0.0, |acc, &id| acc + yp[id])
                / len)
                .powi(2);
    }

    variance
}
