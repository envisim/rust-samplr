use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::kd_tree::{Node, Searcher};
use envisim_utils::matrix::{OperateMatrix, RefMatrix};
use envisim_utils::probability::Probabilities;
use envisim_utils::utils::{sum, usize_to_f64};

#[inline]
pub fn estimate(y_values: &[f64], probabilities: &[f64]) -> Result<f64, SamplingError> {
    InputError::check_lengths(y_values, probabilities).and(Probabilities::check(probabilities))?;

    Ok(y_values
        .iter()
        .zip(probabilities.iter())
        .fold(0.0, |acc, (&y, &p)| acc + y / p))
}

#[inline]
pub fn ratio(
    y_values: &[f64],
    x_values: &[f64],
    probabilities: &[f64],
    x_total: f64,
) -> Result<f64, SamplingError> {
    InputError::check_nonnegative(x_total)?;
    Ok(estimate(y_values, probabilities)? / estimate(x_values, probabilities)? * x_total)
}

pub fn variance(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &RefMatrix,
) -> Result<f64, SamplingError> {
    let sample_size = y_values.len();
    InputError::check_lengths(y_values, probabilities)
        .and(InputError::check_sizes(
            sample_size,
            probabilities_second_order.nrow(),
        ))
        .and(InputError::check_sizes(
            sample_size,
            probabilities_second_order.ncol(),
        ))
        .and(Probabilities::check(probabilities))
        .and(Probabilities::check(probabilities_second_order.data()))?;

    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let y_pi = y_values[i] / probabilities[i];
        variance += y_pi.powi(2) * (1.0 - probabilities[i]);

        for j in (i + 1)..sample_size {
            variance += 2.0 * y_pi * y_values[j] / probabilities[j]
                * (1.0 - probabilities[i] * probabilities[j] / probabilities_second_order[(i, j)]);
        }
    }

    Ok(variance)
}

pub fn syg_variance(
    y_values: &[f64],
    probabilities: &[f64],
    probabilities_second_order: &RefMatrix,
) -> Result<f64, SamplingError> {
    let sample_size = y_values.len();
    InputError::check_lengths(y_values, probabilities)
        .and(InputError::check_sizes(
            sample_size,
            probabilities_second_order.nrow(),
        ))
        .and(InputError::check_sizes(
            sample_size,
            probabilities_second_order.ncol(),
        ))
        .and(Probabilities::check(probabilities))
        .and(Probabilities::check(probabilities_second_order.data()))?;

    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let y_pi = y_values[i] / probabilities[i];

        for j in (i + 1)..sample_size {
            variance -= (y_pi - y_values[j] / probabilities[j]).powi(2)
                * (1.0 - probabilities[i] * probabilities[j] / probabilities_second_order[(i, j)]);
        }
    }

    Ok(variance)
}

pub fn deville_variance(y_values: &[f64], probabilities: &[f64]) -> Result<f64, SamplingError> {
    InputError::check_lengths(y_values, probabilities).and(Probabilities::check(probabilities))?;

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

    Ok(1.0 / (1.0 - sak2) * dsum)
}

pub fn local_mean_variance(
    y_values: &[f64],
    probabilities: &[f64],
    auxilliaries: &RefMatrix,
    n_neighbours: usize,
    bucket_size: usize,
) -> Result<f64, SamplingError> {
    let sample_size = y_values.len();
    InputError::check_lengths(y_values, probabilities)
        .and(InputError::check_sizes(sample_size, auxilliaries.nrow()))
        .and(Probabilities::check(probabilities))?;

    if n_neighbours <= 1 {
        return Err(SamplingError::from(InputError::General(
            "n_neighbours must be > 1".to_string(),
        )));
    }

    let tree = {
        let mut units: Vec<usize> = (0..sample_size).collect();
        Node::new_midpoint_slide(bucket_size, auxilliaries, &mut units)?
    };
    let mut searcher = Searcher::new(&tree, n_neighbours)?;

    let yp: Vec<f64> = y_values
        .iter()
        .zip(probabilities.iter())
        .map(|(&y, &p)| y / p)
        .collect();
    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        searcher
            .find_neighbours_of_iter(&tree, &mut auxilliaries.row_iter(i))
            .unwrap();
        let len = usize_to_f64(searcher.neighbours().len());
        variance += len / (len - 1.0)
            * (searcher
                .neighbours()
                .iter()
                .fold(0.0, |acc, &id| acc + yp[id])
                / len)
                .powi(2);
    }

    Ok(variance)
}
