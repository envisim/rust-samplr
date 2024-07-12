use super::Probabilities;
use crate::utils::usize_to_f64;

pub fn pps_from_slice(arr: &[f64]) -> Probabilities {
    if arr.len() == 0 {
        return Probabilities::new(0, 0.0);
    }

    let mut sum: f64 = 0.0;

    for &x in arr {
        assert!(x > 0.0);
        sum += x;
    }

    Probabilities::with_values(&arr.iter().map(|&x| x / sum).collect::<Vec<f64>>())
}

pub fn pips_from_slice(arr: &[f64], sample_size: usize) -> Probabilities {
    if arr.len() == 0 {
        return Probabilities::new(0, 0.0);
    }

    if arr.len() < sample_size {
        return Probabilities::new(arr.len(), 1.0);
    }

    assert!(arr.iter().all(|&x| x > 0.0));

    let mut n = usize_to_f64(sample_size);

    let mut pips = Probabilities::new(arr.len(), 0.0);
    let mut failed: bool = true;

    while failed && n > 0.0 {
        failed = false;
        let sum: f64 = arr
            .iter()
            .enumerate()
            .filter(|(i, _)| pips[*i] < 1.0)
            .fold(0.0, |acc, (_, &x)| acc + x);
        let curr_n = n;

        arr.iter().enumerate().for_each(|(i, &x)| {
            if pips[i] >= 1.0 {
                return;
            }

            let p = (x * curr_n) / sum;
            pips[i] = p.min(1.0);

            if p >= 1.0 {
                n -= 1.0;

                if !failed && p > 1.0 {
                    failed = true;
                }
            }
        });
    }

    pips
}
