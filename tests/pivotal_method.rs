use envisim_samplr::pivotal_method::*;
mod test_utils;
use test_utils::*;

#[test]
fn test_spm() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.spm(&mut rng), &options, 1e-2, 12000);
}

#[test]
fn test_rpm() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.rpm(&mut rng), &options, 1e-2, 12000);
}

#[test]
fn test_lpm1() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.lpm_1(&mut rng).unwrap(), &options, 0.015, 12000);
}

#[test]
fn test_lpm1s() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.lpm_1s(&mut rng).unwrap(), &options, 0.015, 12000);
}

#[test]
fn test_lpm2() {
    let mut rng = rng();
    let options = options_unequal();
    test_wor(|| options.lpm_2(&mut rng).unwrap(), &options, 0.015, 12000);
}

#[test]
fn test_hlpm2() {
    let mut rng = rng();
    let options = options_unequal();

    test_wor(
        || {
            let grps: [usize; 2] = [1, 4];
            let s = hierarchical_lpm_2(&mut rng, &options, &grps).unwrap();
            assert_eq!(s.len(), 2);
            assert!(grps.iter().enumerate().all(|(i, &g)| { s[i].len() == g }));
            s.into_iter().flatten().collect::<Vec<usize>>()
        },
        &options,
        0.015,
        10000,
    );

    test_wor(
        || {
            let grps: [usize; 3] = [1, 3, 1];
            let s = hierarchical_lpm_2(&mut rng, &options, &grps).unwrap();
            assert_eq!(s.len(), 3);
            assert!(grps.iter().enumerate().all(|(i, &g)| { s[i].len() == g }));
            s.into_iter().flatten().collect::<Vec<usize>>()
        },
        &options,
        0.015,
        10000,
    );
}
