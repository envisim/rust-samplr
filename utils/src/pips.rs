use crate::utils::usize_to_f64;

pub fn pps_from_slice(arr: &[f64]) -> Vec<f64> {
    if arr.len() == 0 {
        return Vec::<f64>::new();
    }

    let mut sum: f64 = 0.0;

    for &x in arr {
        assert!(x > 0.0);
        sum += x;
    }

    arr.iter().map(|&x| x / sum).collect()
}

pub fn pips_from_slice(arr: &[f64], sample_size: usize) -> Vec<f64> {
    if arr.len() == 0 {
        return Vec::<f64>::new();
    }

    if arr.len() < sample_size {
        return vec![1.0f64; arr.len()];
    }

    assert!(arr.iter().all(|&x| x > 0.0));

    let mut n = usize_to_f64(sample_size);

    let mut pips = vec![0.0f64; arr.len()];
    let mut failed: bool = true;

    while failed && n > 0.0 {
        failed = false;
        let sum: f64 = arr
            .iter()
            .enumerate()
            .filter(|(i, _)| pips[*i] < 1.0)
            .fold(0.0, |acc, (_, &x)| acc + x);

        arr.iter().enumerate().for_each(|(i, &x)| {
            if pips[i] >= 1.0 {
                return;
            }

            let p = (x * n) / sum;
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
