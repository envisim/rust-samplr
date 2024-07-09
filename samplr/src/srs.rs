use envisim_samplr_utils::generate_random::GenerateRandom;

pub struct SrsWor {}
impl SrsWor {
    pub fn draw<R: GenerateRandom>(
        rand: &R,
        sample_size: usize,
        population_size: usize,
    ) -> Vec<usize> {
        assert!(
            sample_size <= population_size,
            "sample_size {} must not be larger than population_size {}",
            sample_size,
            population_size,
        );

        let mut sample = Vec::<usize>::with_capacity(sample_size);

        for i in 0..population_size {
            if rand.random_usize_scale(population_size - i) < sample_size - sample.len() {
                sample.push(i);
            }
        }

        sample
    }
}

pub struct SrsWr {}
impl SrsWr {
    pub fn draw<R: GenerateRandom>(
        rand: &R,
        sample_size: usize,
        population_size: usize,
    ) -> Vec<usize> {
        assert!(
            sample_size <= population_size,
            "sample_size {} must not be larger than population_size {}",
            sample_size,
            population_size,
        );

        let mut sample = Vec::<usize>::with_capacity(sample_size);

        for _ in 0..sample_size {
            sample.push(rand.random_usize_scale(population_size));
        }

        sample.sort();
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_samplr_utils::generate_random::StaticRandom;

    // const RAND00: StaticRandom = StaticRandom::new(0.0);
    const RAND01: StaticRandom = StaticRandom::new(0.1);
    // const RAND99: StaticRandom = StaticRandom::new(0.999);

    #[test]
    fn test_srs_wor() {
        assert_eq!(SrsWor::draw(&RAND01, 5, 10), [0, 1, 2, 3, 4]);
        assert_eq!(SrsWor::draw(&RAND01, 2, 10), [0, 1]);
        assert_eq!(SrsWor::draw(&RAND01, 1, 10), [1]);
    }

    #[test]
    fn test_srs_wr() {
        assert_eq!(SrsWr::draw(&RAND01, 5, 10), [1, 1, 1, 1, 1]);
        assert_eq!(SrsWr::draw(&RAND01, 2, 10), [1, 1]);
        assert_eq!(SrsWr::draw(&RAND01, 1, 10), [1]);
    }
}
