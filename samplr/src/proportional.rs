use crate::macros::assert_delta;
use envisim_samplr_utils::probability::Probabilities;
use envisim_samplr_utils::random_generator::RandomGenerator;
use envisim_samplr_utils::utils::sum;

pub fn draw_probabilities_sample<R>(
    rand: &mut R,
    probabilities: &[f64],
    eps: f64,
    n: usize,
) -> Vec<usize>
where
    R: RandomGenerator,
{
    assert!(Probabilities::check(probabilities));
    assert_delta!(sum(probabilities), 1.0, eps);

    if n == 0 {
        return vec![];
    }

    let mut rvs = Vec::<f64>::with_capacity(n);

    for _ in 0..n {
        rvs.push(rand.rf64());
    }

    rvs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sample = Vec::<usize>::with_capacity(n);
    let mut psum: f64 = 0.0;
    let mut rv_iter = rvs.iter();
    let mut rv = *rv_iter.next().unwrap();

    // Add units for which rv is in [psum, psum+p)
    // Go up one p when psum+p < rv
    // Go up one rv when sample has been pushed
    'outer: for (id, &p) in probabilities.iter().enumerate() {
        loop {
            if psum + p <= rv {
                psum += p;
                break;
            }

            if rv < psum + p {
                sample.push(id);

                match rv_iter.next() {
                    Some(v) => {
                        rv = *v;
                        continue;
                    }
                    _ => break 'outer,
                }
            }
        }
    }

    sample
}
