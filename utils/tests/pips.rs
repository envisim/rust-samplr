use envisim_test_utils::*;
use envisim_utils::probability::*;

#[test]
fn pps() {
    let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
    let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];

    assert_fvec(
        pps_from_slice(&dt1).unwrap().data(),
        &vec![0.1, 0.2, 0.3, 0.4],
    );

    assert!(pps_from_slice(&dt2).is_err());
}

#[test]
fn pips() {
    let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
    let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];
    let dt3 = vec![1.0f64, 1.0, 1.0, 7.0];

    assert_fvec(
        pips_from_slice(&dt1, 2).unwrap().data(),
        &vec![0.2, 0.4, 0.6, 0.8],
    );

    assert!(pips_from_slice(&dt2, 2).is_err());

    assert_fvec(
        pips_from_slice(&dt3, 2).unwrap().data(),
        &vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0],
    );
}
