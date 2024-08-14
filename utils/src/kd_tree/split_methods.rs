use crate::matrix::{OperateMatrix, RefMatrix};

// (unit, variable, value)
#[derive(Clone, Debug)]
pub struct Split {
    pub unit: usize,
    pub dimension: usize,
    pub value: f64,
}
pub type FindSplit = fn(&[f64], &[f64], &RefMatrix, &mut [usize]) -> Option<Split>;

pub fn midpoint_slide(
    min_border: &[f64],
    max_border: &[f64],
    data: &RefMatrix,
    units: &mut [usize],
) -> Option<Split> {
    assert!(min_border.len() == max_border.len());
    assert!(data.ncol() == min_border.len());

    let (dimension, spread): (usize, f64) = min_border
        .iter()
        .zip(max_border.iter())
        .map(|(&min, &max)| max - min)
        .enumerate()
        .reduce(|(dim, max_width), (i, width)| {
            if width > max_width {
                (i, width)
            } else {
                (dim, max_width)
            }
        })
        .unwrap();

    if spread <= f64::EPSILON {
        return None;
    }

    let split_value = spread * 0.5 + min_border[dimension];

    let mut split_unit: usize = 0;
    let mut r: usize = units.len();

    // Sort units so that we have
    // x <= value is in range [0, l)
    // x > value is in range [r, n)
    while split_unit < r {
        let lvalue = data[(units[split_unit], dimension)];
        if lvalue <= split_value {
            split_unit += 1;
        } else {
            r -= 1;
            units.swap(split_unit, r);
        }
    }

    if split_unit == 0 || r == units.len() {
        return None;
    }

    Some(Split {
        unit: split_unit,
        dimension,
        value: split_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midpoint_slide() {
        let v = vec![0.0, 1.0, 2.0, 13.0];
        let m = RefMatrix::new(&v, 4);
        let split =
            super::midpoint_slide(&vec![0.0], &vec![13.0], &m, &mut vec![0, 1, 2, 3]).unwrap();
        assert_eq!(split.unit, 3);
        assert_eq!(split.dimension, 0);
        assert_eq!(split.value, 6.5);

        let v = vec![0.0, 1.0, 2.0, 13.0, 0.0, 10.0, 20.0, 30.0];
        let m = RefMatrix::new(&v, 4);
        let split = super::midpoint_slide(
            &vec![0.0, 0.0],
            &vec![13.0, 30.0],
            &m,
            &mut vec![0, 1, 2, 3],
        )
        .unwrap();
        assert_eq!(split.unit, 2);
        assert_eq!(split.dimension, 1);
        assert_eq!(split.value, 15.0);

        let v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let m = RefMatrix::new(&v, 3);
        let split = super::midpoint_slide(&vec![0.0, 1.0], &vec![0.0, 1.0], &m, &mut vec![0, 1, 2]);
        assert!(split.is_none());
    }
}
