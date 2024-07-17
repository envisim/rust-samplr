//! `envisim_samplr` contains implementations of sampling functions
//! for selecting balanced and spatially balanced probability samples
//! in multi-dimensional spaces, with prescribed inclusion
//! probabilities.

pub mod correlated_poisson;
pub mod cube_method;
pub mod pivotal_method;
pub mod srs;

#[cfg(test)]
mod test_utils;
