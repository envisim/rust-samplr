// Copyright (C) 2026 Wilmer Prentius.
//
// This progra6 is free software: yo can redistribute it and/or modify it under the terms of the
// GNU Affero General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with this
// program. If not, see <https://www.gnu.org/licenses/>.

//! Functions for calculating probabilities proportional to size

use std::num::NonZeroUsize;

use envisim_utils::probabilities::{
    Probability,
    ProbabilityContext,
    ProbabilitySet,
    ProbabilityStore,
};
use envisim_utils::utils::{
    ConstructableDataView,
    Epsilon,
    Number,
};
use num_traits::ToPrimitive;
use thiserror::Error;

/// Construct draw probabilities proportional to size.
/// Given an array of positive values, returns draw probabilities proportional to size.
///
/// # Errors
/// Returns an error if any value is non-positive.
/// # Panics
/// Panics if values in `auxiliaries` cannot be converted to `f64`.
#[inline]
pub fn pps<D>(
    auxiliaries: D,
) -> Result<ProbabilitySet<D::ConstructableContainer<Probability<f64>>, f64>, PipsError>
where
    D: ConstructableDataView<Value: Number>,
{
    if auxiliaries.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    }

    let mut sum = 0.0;

    for x in auxiliaries.values() {
        if !x.is_pos_finite() {
            return Err(PipsError::InvalidAuxiliary);
        }
        sum += x.to_f64().expect("aux converts to f64");
    }

    let pps = auxiliaries.iter_map(|(i, v)| {
        (
            i,
            Probability::new_real(v.to_f64().expect("aux converts to f64") / sum)
                .expect("v <= sum"),
        )
    });
    Ok(ProbabilitySet::new(
        pps,
        ProbabilityContext::new_real(Epsilon::default()),
    ))
}

/// Inclusion probabilities proportional to size (approximate).
/// Given an array of positive values, returns the inclusion probabilities proportional to size.
///
/// The caluclations are done by iteratively rescaling the inclusion probabilities.
///
/// # Errors
/// Returns an error if any value is non-positive.
/// # Panics
/// Panics if values in `auxiliaries` cannot be converted to `f64`.
#[inline]
pub fn pips<D>(
    auxiliaries: D,
    sample_size: NonZeroUsize,
) -> Result<ProbabilitySet<D::ConstructableContainer<Probability<f64>>, f64>, PipsError>
where
    D: ConstructableDataView<Value: Number>,
{
    if auxiliaries.is_empty() {
        return Err(PipsError::NoAuxiliaries);
    } else if auxiliaries.len() < sample_size.get() {
        return Err(PipsError::InvalidSampleSize);
    } else if auxiliaries.len() == sample_size.get() {
        let pips =
            auxiliaries.iter_map(|(i, _)| (i, Probability::new_real(1.0).expect("1.0 in [0,1]")));
        return Ok(ProbabilitySet::new(
            pips,
            ProbabilityContext::new_real(Epsilon::default()),
        ));
    } else if auxiliaries.values().any(|x| !x.is_pos_finite()) {
        return Err(PipsError::InvalidAuxiliary);
    }

    let mut n = sample_size
        .get()
        .to_f64()
        .expect("sample size converts to f64");
    let mut pips = {
        let p =
            auxiliaries.iter_map(|(i, _)| (i, Probability::new_real(0.0).expect("0.0 in [0,1]")));
        ProbabilitySet::new(p, ProbabilityContext::new_real(Epsilon::default()))
    };
    let mut failed = true;

    while failed && n > 0.0 {
        failed = false;
        let sum = auxiliaries
            .entries()
            .filter_map(|(id, v)| (!pips.is_full(id).expect("id to exist")).then_some(*v))
            .sum::<D::Value>()
            .to_f64()
            .expect("aux converts to f64");
        let curr_n = n;

        for (id, x) in auxiliaries.entries() {
            if pips.is_full(id).expect("i to exist") {
                // cannot filter b/c immutable borrow
                continue;
            }
            let p0 = x.to_f64().expect("aux converts to f64") * curr_n / sum;
            let p = Probability::new_real(p0.min(1.0)).expect("p in [0,1]");

            pips.set(id, p).expect("i to exist");
            if p.is_full(pips.ctx()) {
                n -= 1.0;
                if p0 > 1.0 {
                    failed = true;
                }
            }
        }
    }

    Ok(pips)
}

/// Pips related errors
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum PipsError {
    /// Auxiliaries must be positive valued
    #[error("auxiliaries must be positive")]
    InvalidAuxiliary,
    /// Sample size must be in (0, N]
    #[error("sample size must be positive and less than pop size")]
    InvalidSampleSize,
    /// Auxiliaries must be provided
    #[error("slice contains no auxiliaries")]
    NoAuxiliaries,
}

#[cfg(test)]
mod tests {
    use envisim_utils::probabilities::ProbabilityStoreToRaw;
    use envisim_utils::test_utils::*;

    use super::*;

    #[test]
    fn test_pps() {
        let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
        let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];

        let res = pps(&dt1).unwrap().to_raw();
        assert_vec!(res.slice(), &[0.1, 0.2, 0.3, 0.4]);
        assert!(pps(&dt2).is_err());
    }

    #[test]
    fn test_pips() {
        let ss = NonZeroUsize::new(2).unwrap();
        let dt1 = vec![1.0f64, 2.0, 3.0, 4.0];
        let dt2 = vec![-1.0f64, 2.0, 3.0, 4.0];
        let dt3 = vec![1.0f64, 1.0, 1.0, 7.0];

        let pips1 = pips(&dt1, ss).unwrap();
        assert_vec!(pips1.to_raw(), [0.2, 0.4, 0.6, 0.8]);

        assert!(pips(&dt2, ss).is_err());

        let pips3 = pips(&dt3, ss).unwrap();
        assert_vec!(pips3.to_raw(), [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
    }
}
