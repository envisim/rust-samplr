use crate::matrix::OperateMatrix;

// (unit, variable, value)
pub type FindSplit = fn(&[f64], &[f64], &dyn OperateMatrix, &mut [usize]) -> (usize, usize, f64);

pub fn midpoint_slide(
    min_border: &[f64],
    max_border: &[f64],
    data: &dyn OperateMatrix,
    units: &mut [usize],
) -> (usize, usize, f64) {
    assert!(min_border.len() == max_border.len());
    assert!(data.ncol() == min_border.len());

    let mut split_i: usize = 0;
    let mut split_dimension: usize = 0;
    let mut split_value: f64 = 0.0;

    let mut spread: f64 = 0.0;

    unsafe {
        for k in 0usize..data.ncol() {
            let width = max_border.get_unchecked(k) - min_border.get_unchecked(k);
            if width > spread {
                spread = width;
                split_dimension = k;
                split_value = width * 0.5 + min_border.get_unchecked(k);
            }
        }
    }

    if spread <= f64::EPSILON {
        return (0, 0, 0.0);
    }

    let mut r: usize = units.len();

    // Sort units so that we have
    // x <= value is in range [0, l)
    // x > value is in range [r, n)
    while split_i < r {
        let lvalue = data[(units[split_i], split_dimension)];
        if lvalue <= split_value {
            split_i += 1;
        } else {
            r -= 1;
            units.swap(split_i, r);
        }
    }

    if split_i == 0 || r == units.len() {
        return (0, 0, 0.0);
    }

    (split_i, split_dimension, split_value)
}
