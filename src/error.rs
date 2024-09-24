use envisim_utils::kd_tree::NodeError;
use envisim_utils::{IndicesError, InputError};
use std::num::NonZeroUsize;

/// Sampling related error types
#[non_exhaustive]
#[derive(Debug)]
pub enum SamplingError {
    Indices(IndicesError),
    Input(InputError),
    Node(NodeError),
    // max iterations reached
    MaxIterations(NonZeroUsize),
}

impl std::error::Error for SamplingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            SamplingError::Indices(ref err) => Some(err),
            SamplingError::Input(ref err) => Some(err),
            SamplingError::Node(ref err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            SamplingError::Indices(ref err) => err.fmt(f),
            SamplingError::Input(ref err) => err.fmt(f),
            SamplingError::Node(ref err) => err.fmt(f),
            SamplingError::MaxIterations(max_iter) => {
                write!(f, "max iterations ({max_iter}) reached")
            }
        }
    }
}

impl From<IndicesError> for SamplingError {
    fn from(err: IndicesError) -> SamplingError {
        SamplingError::Indices(err)
    }
}
impl From<InputError> for SamplingError {
    fn from(err: InputError) -> SamplingError {
        SamplingError::Input(err)
    }
}
impl From<NodeError> for SamplingError {
    fn from(err: NodeError) -> SamplingError {
        SamplingError::Node(err)
    }
}
