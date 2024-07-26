use envisim_utils::error::{InputError, SamplingError};
use envisim_utils::matrix::{OperateMatrix, RefMatrix};

#[inline]
fn inclusions_check(inclusions: &[f64]) -> Result<(), InputError> {
    inclusions.iter().try_for_each(|&inc| {
        InputError::check_nan(inc)
            .and(InputError::check_nonnegative(inc))
            .and(InputError::check_integer(inc))
    })
}

#[inline]
fn expected_inclusions_check(expected: &[f64]) -> Result<(), InputError> {
    expected
        .iter()
        .try_for_each(|&mu| InputError::check_nan(mu).and(InputError::check_positive(mu)))
}

#[inline]
pub fn estimate(
    y_values: &[f64],
    expected: &[f64],
    inclusions: &[f64],
) -> Result<f64, SamplingError> {
    InputError::check_lengths(y_values, expected)
        .and(InputError::check_lengths(y_values, inclusions))
        .and(expected_inclusions_check(expected))
        .and(inclusions_check(inclusions))
        .map_err(SamplingError::from)?;

    Ok(y_values
        .iter()
        .zip(expected.iter())
        .zip(inclusions.iter())
        .fold(0.0, |acc, ((&y, &mu), &inc)| acc + y / mu * inc))
}

pub fn variance(
    y_values: &[f64],
    expected: &[f64],
    inclusions: &[f64],
    expected_second_order: &RefMatrix,
) -> Result<f64, SamplingError> {
    let sample_size = y_values.len();
    InputError::check_lengths(y_values, expected)
        .and(InputError::check_lengths(y_values, inclusions))
        .and(InputError::check_sizes(
            sample_size,
            expected_second_order.nrow(),
        ))
        .and(InputError::check_sizes(
            sample_size,
            expected_second_order.ncol(),
        ))
        .and(expected_inclusions_check(expected))
        .and(inclusions_check(inclusions))
        .and(expected_inclusions_check(expected_second_order.data()))
        .map_err(SamplingError::from)?;

    let mut variance: f64 = 0.0;

    for i in 0..sample_size {
        let y_mu_inc = y_values[i] / expected[i] * inclusions[i];
        variance += y_mu_inc.powi(2) * (1.0 - expected[i].powi(2) / expected_second_order[(i, i)]);

        for j in (i + 1)..sample_size {
            variance += 2.0 * y_mu_inc * y_values[j] / expected[j]
                * inclusions[j]
                * (1.0 - expected[i] * expected[j] / expected_second_order[(i, j)]);
        }
    }

    Ok(variance)
}
