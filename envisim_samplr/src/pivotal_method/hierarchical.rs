// Copyright (C) 2026 Wilmer Prentius.
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

use envisim_utils::kd_tree::searcher::NearestNeighbourSearcher;
use envisim_utils::probabilities::ProbabilityStore;
use envisim_utils::random::RandomNumberGenerator;
use envisim_utils::sample_controller::SampleController;
use envisim_utils::sampling_options::{
    ProbabilitySpec,
    SamplingOptions,
};
use envisim_utils::spatial::{
    Number,
    PointSet,
};
use envisim_utils::utils::usize_to_f64;
use rustc_hash::FxHashSet;

use super::runner::PivotalRunner;
use super::spatial::LocalStrategy2;
use crate::error::{
    SamplingError,
    SamplingResult,
};

/// Draw a sample using the hierarchical local pivotal method 2.
/// The sample is spatially balanced on the provided auxilliary variables in `data`.
/// Selects an initial sample using [`lpm_2`], and splits this sample into subsamples of given
/// `sizes`, using successive, hierarchical selection with `lpm_2`.
/// `sizes` must sum to the sum of `probabilities`.
///
/// # Examples
/// ```
/// # use envisim_samplr::*;
/// # use envisim_samplr::pivotal_method::hierarchical_lpm_2;
/// use envisim_utils::random::*;
/// use envisim_utils::matrix::Matrix;
///
/// let mut rng = SmallRng::from_os_rng();
/// let p = [0.2, 0.25, 0.35, 0.4, 0.5, 0.5, 0.55, 0.65, 0.7, 0.9];
/// let m = Matrix::from_vec(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 10).unwrap();
/// let options = SamplingOptions::new(&p)?.set_spreading(m)?;
/// let sizes = [3, 2];
/// let s = hierarchical_lpm_2(&mut rng, &options, &sizes)?;
///
/// assert_eq!(s.len(), 2);
/// # Ok::<(), SamplingError>(())
/// ```
///
/// # References
/// Grafström, A., Lundström, N. L., & Schelin, L. (2012).
/// Spatially balanced sampling through the pivotal method.
/// Biometrics, 68(2), 514-520.
/// <https://doi.org/10.1111/j.1541-0420.2011.01699.x>
pub fn hierarchical_lpm_2<'a, R, PS, SOP, M, N>(
    rng: &mut R,
    options: &'a SamplingOptions<'a, PS, SOP, M>,
    sizes: &[usize],
) -> SamplingResult<Vec<Vec<usize>>>
where
    R: RandomNumberGenerator,
    PS: ProbabilitySpec,
    SOP: PointSet<N>,
    N: Number,
{
    // Check validity of probabilities and sizes
    let sizes_sum = sizes.iter().sum();
    let sample_size = options.probabilities().sample_size();
    let psum = options.probabilities().sample_size_f64();
    if sample_size != sizes_sum || (psum.round() - psum).abs() > options.eps() {
        return Err(SamplingError::IncorrectStratification);
    }

    // Cannot use ::new, controller needs to be float
    let mut pm = {
        let controller = options.to_spreading_controller_float()?;
        let searcher = NearestNeighbourSearcher::new(controller.tree().data());
        PivotalRunner {
            controller,
            strategy: LocalStrategy2 { searcher },
        }
    };
    pm.run(rng);

    if sizes.len() == 1 {
        return Ok(vec![pm.controller.sample_mut().sort_to_vec()]);
    }

    let mut return_sample = Vec::<Vec<usize>>::with_capacity(sizes.len());
    let mut main_sample: FxHashSet<usize> = pm.controller.sample().get().iter().cloned().collect();

    for &size in sizes[0..sizes.len() - 1].iter() {
        assert!(pm.controller.indices().is_empty());

        if size == 0 {
            return_sample.push(vec![]);
        }

        pm.controller.sample_mut().clear();

        let prob = usize_to_f64(size) / usize_to_f64(main_sample.len());

        // Reset probs and add to indices/tree
        for id in 0..pm.controller.population_size() {
            if main_sample.contains(&id) {
                pm.controller.probabilities_mut().set(id, prob);
                pm.controller.indices_mut().insert(id).unwrap();
                pm.controller.tree_mut().insert_unit(id).unwrap();
            } else {
                pm.controller.probabilities_mut().set_zero(id);
            }
        }

        pm.run(rng);

        let s = pm.controller.sample_mut().sort_to_vec();
        s.iter().for_each(|id| {
            main_sample.remove(id);
        });

        return_sample.push(s);
    }

    let mut s: Vec<usize> = main_sample.into_iter().collect();
    s.sort_unstable();
    return_sample.push(s);

    Ok(return_sample)
}
