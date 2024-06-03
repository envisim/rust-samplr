use envisim_samplr_utils::uniform_random::{discrete_uniform_u, RandomGenerator};

pub fn srs_wor(rand: &RandomGenerator, sample_size: usize, population_size: usize) -> Vec<usize> {
    assert!(
        sample_size <= population_size,
        "sample_size {} must not be larger than population_size {}",
        sample_size,
        population_size,
    );

    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for i in 0..population_size {
        if discrete_uniform_u(rand, population_size - i) < sample_size - sample.len() {
            sample.push(i);
        }
    }

    sample
}

pub fn srs_wr(rand: &RandomGenerator, sample_size: usize, population_size: usize) -> Vec<usize> {
    assert!(
        sample_size <= population_size,
        "sample_size {} must not be larger than population_size {}",
        sample_size,
        population_size,
    );

    let mut sample = Vec::<usize>::with_capacity(sample_size);

    for _ in 0..sample_size {
        sample.push(discrete_uniform_u(rand, population_size));
    }

    sample.sort();
    sample
}

#[cfg(test)]
mod tests {
    use super::*;

    // const RAND00: RandomGenerator = || 0.0;
    const RAND01: RandomGenerator = || 0.1;
    // const RAND99: RandomGenerator = || 0.999;

    #[test]
    fn test_srs_wor() {
        assert_eq!(srs_wor(&RAND01, 5, 10), [0, 1, 2, 3, 4]);
        assert_eq!(srs_wor(&RAND01, 2, 10), [0, 1]);
        assert_eq!(srs_wor(&RAND01, 1, 10), [1]);
    }

    #[test]
    fn test_srs_wr() {
        assert_eq!(srs_wr(&RAND01, 5, 10), [1, 1, 1, 1, 1]);
        assert_eq!(srs_wr(&RAND01, 2, 10), [1, 1]);
        assert_eq!(srs_wr(&RAND01, 1, 10), [1]);
    }
}
