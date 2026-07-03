pub use crate::matrix::Dimensions;
use crate::matrix::MatrixRef;
use crate::sampling_options::{
    BalancingOptions,
    EqualProbabilityOptions,
    RealUnequalProbabilityOptions,
    SamplingOptions,
    SpreadingOptions,
    UnequalProbabilityOptions,
};

// DISTS:
//        1     2     3     4     5     6     7     8     9    10
// 1  0.000 0.110 0.571 0.666 0.568 0.696 0.850 0.880 0.403 0.606
// 2  0.110 0.000 0.548 0.575 0.617 0.616 0.788 0.865 0.328 0.675
// 3  0.571 0.548 0.000 0.452 0.380 0.376 0.373 0.317 0.312 0.519
// 4  0.666 0.575 0.452 0.000 0.805 0.114 0.336 0.656 0.279 0.933
// 5  0.568 0.617 0.380 0.805 0.000 0.747 0.745 0.510 0.578 0.140
// 6  0.696 0.616 0.376 0.114 0.747 0.000 0.225 0.548 0.294 0.881
// 7  0.850 0.788 0.373 0.336 0.745 0.225 0.000 0.395 0.463 0.885
// 8  0.880 0.865 0.317 0.656 0.510 0.548 0.395 0.000 0.613 0.636
// 9  0.403 0.328 0.312 0.279 0.578 0.294 0.463 0.613 0.000 0.692
// 10 0.606 0.675 0.519 0.933 0.140 0.881 0.885 0.636 0.692 0.000

// NEIGHBOURS (row's closest neighbours):
// 0:  0 1 8 4 2 9 3 5 6 7
// 1:  1 0 8 2 3 5 4 9 6 7
// 2:  2 8 7 6 5 4 3 9 1 0
// 3:  3 5 8 6 2 1 7 0 4 9
// 4:  4 9 2 7 0 8 1 6 5 3
// 5:  5 3 6 8 2 7 1 0 4 9
// 6:  6 5 3 2 7 8 4 1 0 9
// 7:  7 2 6 4 5 8 9 3 1 0
// 8:  8 3 5 2 1 0 6 4 7 9
// 9:  9 4 2 0 7 1 8 5 6 3

pub struct Data10();
impl Data10 {
    pub const PROB_U: [f64; 10] = [0.20, 0.25, 0.35, 0.40, 0.50, 0.50, 0.55, 0.65, 0.70, 0.90];
    pub const PROB_E: [f64; 10] = [0.2f64; 10];
    pub const DATA_2: [f64; 20] = [
        0.266, 0.372, 0.573, 0.908, 0.202, 0.898, 0.945, 0.661, 0.629, 0.062, //
        0.206, 0.177, 0.687, 0.384, 0.770, 0.498, 0.718, 0.992, 0.380, 0.777, //
    ];
    pub const BDATA_1: [f64; 10] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
    ];
    pub const BDATA_UP1: [f64; 20] = [
        0.20, 0.25, 0.35, 0.40, 0.50, 0.50, 0.55, 0.65, 0.70, 0.90, //
        0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, //
    ];
    pub const BDATA_EP1: [f64; 20] = [
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, //
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
    ];

    #[inline]
    pub fn prob_u() -> UnequalProbabilityOptions<RealUnequalProbabilityOptions<&'static [f64]>> {
        UnequalProbabilityOptions::new(Self::PROB_U.as_slice()).unwrap()
    }
    #[inline]
    pub fn prob_e() -> EqualProbabilityOptions { EqualProbabilityOptions::new(10, 2).unwrap() }
    #[inline]
    pub fn matrix() -> MatrixRef<'static, f64> { MatrixRef::new(&Self::DATA_2, 10).unwrap() }
    #[inline]
    pub fn bmatrix_no_p() -> MatrixRef<'static, f64> { MatrixRef::new(&Self::BDATA_1, 10).unwrap() }
    #[inline]
    pub fn bmatrix_up() -> MatrixRef<'static, f64> { MatrixRef::new(&Self::BDATA_UP1, 10).unwrap() }
    #[inline]
    pub fn bmatrix_ep() -> MatrixRef<'static, f64> { MatrixRef::new(&Self::BDATA_EP1, 10).unwrap() }
    #[inline]
    pub fn options_u() -> SamplingOptions<
        UnequalProbabilityOptions<RealUnequalProbabilityOptions<&'static [f64]>>,
        SpreadingOptions<MatrixRef<'static, f64>>,
        BalancingOptions<MatrixRef<'static, f64>>,
    > {
        SamplingOptions::with_spec(Self::prob_u())
            .set_spreading(Self::matrix())
            .unwrap()
            .set_balancing(Self::bmatrix_up())
            .unwrap()
    }
    #[inline]
    pub fn options_e() -> SamplingOptions<
        EqualProbabilityOptions,
        SpreadingOptions<MatrixRef<'static, f64>>,
        BalancingOptions<MatrixRef<'static, f64>>,
    > {
        SamplingOptions::with_spec_equal(Data10::prob_e())
            .set_spreading(Data10::matrix())
            .unwrap()
            .set_balancing(Data10::bmatrix_ep())
            .unwrap()
    }
}
