use crate::indices::IndicesError;
use crate::kd_tree::{NodeError, SearcherError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InputError {
    #[error("{0}")]
    General(String),
    #[error("probability must be in [0.0, 1.0]")]
    InvalidProbability,
    #[error("epsilon must be in [0.0, 1.0)")]
    InvalidEpsilon,
    #[error("index must be less than ({0})")]
    InvalidIndex(usize),
    #[error("value must be positive (0.0, +Inf)")]
    NonpositiveValue,
    #[error("value must be nonnegative [0.0, +Inf)")]
    NegativeValue,
    #[error("value must be integer valued")]
    NotInteger,
    #[error("value must not be NaN")]
    NanValue,
    #[error("sizes does not match, {0} vs {1}")]
    SizeDiff(usize, usize),
    #[error("sample size must not be larger than population size")]
    SampleSize,
    #[error("slice is empty")]
    IsEmpty,
}

impl InputError {
    #[inline]
    pub fn check_empty<T>(a: &[T]) -> Result<(), InputError> {
        if a.is_empty() {
            return Err(InputError::IsEmpty);
        }

        Ok(())
    }

    #[inline]
    pub fn check_lengths<TA, TB>(a: &[TA], b: &[TB]) -> Result<(), InputError> {
        InputError::check_sizes(a.len(), b.len())
    }
    #[inline]
    pub fn check_sizes(a: usize, b: usize) -> Result<(), InputError> {
        if a != b {
            return Err(InputError::SizeDiff(a, b));
        }

        Ok(())
    }
    #[inline]
    pub fn check_sample_size(sample_size: usize, population_size: usize) -> Result<(), InputError> {
        if population_size < sample_size {
            return Err(InputError::SampleSize);
        }

        Ok(())
    }
    #[inline]
    pub fn check_nan(value: f64) -> Result<(), InputError> {
        if value.is_nan() {
            return Err(InputError::NanValue);
        }

        Ok(())
    }
    #[inline]
    pub fn check_positive(value: f64) -> Result<(), InputError> {
        if value <= 0.0 {
            return Err(InputError::NonpositiveValue);
        }

        Ok(())
    }
    #[inline]
    pub fn check_nonnegative(value: f64) -> Result<(), InputError> {
        if value < 0.0 {
            return Err(InputError::NegativeValue);
        }

        Ok(())
    }
    #[inline]
    pub fn check_integer(value: f64) -> Result<(), InputError> {
        if value != value.round() {
            return Err(InputError::NotInteger);
        }

        Ok(())
    }
    #[inline]
    pub fn check_integer_approx(value: f64, eps: f64) -> Result<(), InputError> {
        if (value - eps..value + eps).contains(&value.round()) {
            return Err(InputError::NotInteger);
        }

        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum SamplingError {
    #[error("{0}")]
    General(String),
    #[error(transparent)]
    Indices(#[from] IndicesError),
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Node(#[from] NodeError),
    #[error(transparent)]
    Searcher(#[from] SearcherError),
    #[error("max iterations reached")]
    MaxIterations,
}
