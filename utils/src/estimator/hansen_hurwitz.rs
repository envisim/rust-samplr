use crate::matrix::{OperateMatrix, RefMatrix};

#[inline]
fn inclusions_check(inclusions: &[f64]) -> bool {
    inclusions
        .iter()
        .all(|&inc| 0.0 <= inc && inc == inc.round())
}

#[inline]
fn expected_inclusions_check(expected: &[f64]) -> bool {
    expected.iter().all(|&mu| 0.0 <= mu)
}

#[inline]
pub fn estimate(y_values: &[f64], expected: &[f64], inclusions: &[f64]) -> f64 {
    assert_eq!(y_values.len(), expected.len());
    assert_eq!(y_values.len(), inclusions.len());
    assert!(expected_inclusions_check(expected));
    assert!(inclusions_check(inclusions));

    y_values
        .iter()
        .zip(expected.iter())
        .zip(inclusions.iter())
        .fold(0.0, |acc, ((&y, &mu), &inc)| acc + y / mu * inc)
}

pub fn variance(
    y_values: &[f64],
    expected: &[f64],
    inclusions: &[f64],
    expected_second_order: &RefMatrix,
) -> f64 {
    let sample_size = y_values.len();
    assert_eq!(sample_size, expected.len());
    assert_eq!(sample_size, expected_second_order.nrow());
    assert_eq!(sample_size, expected_second_order.ncol());
    assert!(expected_inclusions_check(expected));
    assert!(expected_inclusions_check(expected_second_order.data()));
    assert!(inclusions_check(inclusions));

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

    variance
}
