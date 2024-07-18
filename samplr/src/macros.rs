macro_rules! assert_delta {
    ($a:expr,$b:expr,$d:expr) => {
        assert!(($a - $b).abs() < $d, "|{} - {}| >= {}", $a, $b, $d);
    };
}

pub(crate) use assert_delta;
