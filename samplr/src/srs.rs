use envisim_samplr_utils::random_generator::RandomGenerator;

pub struct SrsWor {}
impl SrsWor {
    pub fn draw<R>(rand: &R, sample_size: usize, population_size: usize) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        assert!(
            sample_size <= population_size,
            "sample_size {} must not be larger than population_size {}",
            sample_size,
            population_size,
        );

        let mut sample = Vec::<usize>::with_capacity(sample_size);

        for i in 0..population_size {
            if rand.rusize(population_size - i) < sample_size - sample.len() {
                sample.push(i);
            }
        }

        sample
    }
}

pub struct SrsWr {}
impl SrsWr {
    pub fn draw<R>(rand: &R, sample_size: usize, population_size: usize) -> Vec<usize>
    where
        R: RandomGenerator,
    {
        assert!(
            sample_size <= population_size,
            "sample_size {} must not be larger than population_size {}",
            sample_size,
            population_size,
        );

        let mut sample: Vec<usize> = (0..sample_size)
            .map(|_| rand.rusize(population_size))
            .collect();

        sample.sort_unstable();
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use envisim_samplr_utils::random_generator::Constant;

    const RAND01: Constant = Constant::new(0.1);

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
