use envisim_samplr::pivotal_method::*;
use envisim_test_utils::*;
use envisim_utils::matrix::RefMatrix;
use envisim_utils::utils::sum;

mod test_utils;
use test_utils::*;

#[test]
fn test_spm() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| spm(&mut rng, p, EPS), p, 1e-2, 100000);
}

#[test]
fn test_rpm() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;

    test_wor(|| rpm(&mut rng, p, EPS), p, 1e-2, 100000);
}

#[test]
fn test_lpm1() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(
        || lpm_1(&mut rng, p, EPS, &data, NONZERO_2),
        p,
        1e-2,
        100000,
    );
}

#[test]
fn test_lpm1s() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(
        || lpm_1s(&mut rng, p, EPS, &data, NONZERO_2),
        p,
        1e-2,
        100000,
    );
}

#[test]
fn test_lpm2() {
    let mut rng = seeded_rng();
    let p = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    test_wor(
        || lpm_2(&mut rng, p, EPS, &data, NONZERO_2),
        p,
        1e-2,
        100000,
    );
}

#[test]
fn test_hlpm2() {
    let eps = 1e-2;
    let iter = 100000;

    let mut rng = seeded_rng();
    let probs = &PROB_10_U;
    let data = RefMatrix::new(&DATA_10_2, 10);

    {
        let mut sel: Vec<u32> = vec![0; probs.len()];

        for _ in 0..iter {
            let s = hierarchical_lpm_2(&mut rng, probs, EPS, &data, NONZERO_2, &[1, 4]).unwrap();
            assert!(s.len() == 2 && s[0].len() == 1 && s[1].len() == 4);
            s.iter().flatten().for_each(|&id| sel[id] += 1);
        }

        let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
        let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

        if !d.iter().all(|&x| x.abs() < eps) {
            panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(probs), sum(&q));
        }
    }

    {
        let mut sel: Vec<u32> = vec![0; probs.len()];

        for _ in 0..iter {
            let s = hierarchical_lpm_2(&mut rng, probs, EPS, &data, NONZERO_2, &[1, 3, 1]).unwrap();
            assert!(s.len() == 3 && s[0].len() == 1 && s[1].len() == 3 && s[2].len() == 1);
            s.iter().flatten().for_each(|&id| sel[id] += 1);
        }

        let q: Vec<f64> = sel.iter().map(|&s| (s as f64) / (iter as f64)).collect();
        let d: Vec<f64> = probs.iter().zip(q.iter()).map(|(p, r)| p - r).collect();

        if !d.iter().all(|&x| x.abs() < eps) {
            panic!("{d:?} >= {eps}\n(sums: {} vs. {})", sum(probs), sum(&q));
        }
    }
}
