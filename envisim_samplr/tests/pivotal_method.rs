use envisim_samplr::pivotal_method::*;
mod test_utils;
use test_utils::*;

#[test]
fn test_spm() {
    let options = Data10::options_u();
    test_wor(|rng| options.spm(rng), &options, 1e-2, 10000);
}

#[test]
fn test_rpm() {
    let options = Data10::options_u();
    test_wor(|rng| options.rpm(rng), &options, 1e-2, 10000);
}

#[test]
fn test_lpm1() {
    let options = Data10::options_u();
    test_wor(|rng| options.lpm_1(rng).unwrap(), &options, 0.015, 10000);
}

#[test]
fn test_lpm1s() {
    let options = Data10::options_u();
    test_wor(|rng| options.lpm_1s(rng).unwrap(), &options, 0.015, 10000);
}

#[test]
fn test_lpm2() {
    let options = Data10::options_u();
    test_wor(|rng| options.lpm_2(rng).unwrap(), &options, 0.015, 10000);
}

#[test]
fn test_hlpm2() {
    let options = Data10::options_u();

    test_wor(
        |rng| {
            let grps: [usize; 2] = [1, 4];
            let s = hierarchical_lpm_2(rng, &options, &grps).unwrap();
            assert_eq!(s.len(), 2);
            assert!(grps.iter().enumerate().all(|(i, &g)| { s[i].len() == g }));
            s.into_iter().flatten().collect::<Vec<usize>>()
        },
        &options,
        0.015,
        10000,
    );

    test_wor(
        |rng| {
            let grps: [usize; 3] = [1, 3, 1];
            let s = hierarchical_lpm_2(rng, &options, &grps).unwrap();
            assert_eq!(s.len(), 3);
            assert!(grps.iter().enumerate().all(|(i, &g)| { s[i].len() == g }));
            s.into_iter().flatten().collect::<Vec<usize>>()
        },
        &options,
        0.015,
        10000,
    );
}
