pub trait Probability {
    fn in_range(self, eps: Self) -> bool;
    fn is_zero(self, eps: Self) -> bool;
    fn is_one(self, eps: Self) -> bool;
}

impl Probability for f64 {
    #[inline]
    fn in_range(self, eps: f64) -> bool {
        self > eps && self < 1.0 - eps
    }
    #[inline]
    fn is_zero(self, eps: f64) -> bool {
        self <= eps
    }
    #[inline]
    fn is_one(self, eps: f64) -> bool {
        self >= 1.0 - eps
    }
}

impl Probability for i64 {
    #[inline]
    fn in_range(self, n: i64) -> bool {
        self > 0i64 && self < n
    }
    #[inline]
    fn is_zero(self, _n: i64) -> bool {
        self <= 0i64
    }
    #[inline]
    fn is_one(self, n: i64) -> bool {
        self >= n
    }
}
