use crate::SamplingError;
use envisim_utils::kd_tree::{midpoint_slide, FindSplit, Node, TreeBuilder};
use envisim_utils::{InputError, Matrix, Probabilities};
use rand::Rng;
use std::num::NonZeroUsize;

pub struct SampleOptions<'a> {
    // Base
    pub(crate) probabilities: &'a [f64],
    pub(crate) eps: f64,
    pub(crate) max_iterations: NonZeroUsize,

    // Spatially balanced sampling
    pub(crate) auxiliaries: Option<&'a Matrix<'a>>,
    pub(crate) bucket_size: NonZeroUsize,
    pub(crate) split_method: FindSplit,

    // Balanced sampling
    pub(crate) balancing: Option<&'a Matrix<'a>>,

    // Coordinated
    pub(crate) random_values: Option<&'a [f64]>,
}

impl<'a> SampleOptions<'a> {
    #[inline]
    pub fn new(probabilities: &'a [f64]) -> Result<Self, InputError> {
        Probabilities::check(probabilities)?;

        Ok(Self {
            probabilities,
            eps: 1e-12,
            max_iterations: unsafe { NonZeroUsize::new_unchecked(1000) },
            auxiliaries: None,
            bucket_size: unsafe { NonZeroUsize::new_unchecked(40) },
            split_method: midpoint_slide,
            balancing: None,
            random_values: None,
        })
    }
    #[inline]
    pub fn eps(&mut self, eps: f64) -> Result<&mut Self, InputError> {
        self.eps = Probabilities::check_eps(eps)?;
        Ok(self)
    }
    #[inline]
    pub fn max_iterations(
        &mut self,
        max_iterations: NonZeroUsize,
    ) -> Result<&mut Self, InputError> {
        self.max_iterations = max_iterations;
        Ok(self)
    }
    #[inline]
    pub fn auxiliaries(&mut self, auxiliaries: &'a Matrix<'a>) -> Result<&mut Self, InputError> {
        InputError::check_sizes(auxiliaries.nrow(), self.probabilities.len())?;
        self.auxiliaries = Some(auxiliaries);
        Ok(self)
    }
    #[inline]
    pub fn bucket_size(&mut self, bucket_size: NonZeroUsize) -> Result<&mut Self, InputError> {
        self.bucket_size = bucket_size;
        Ok(self)
    }
    #[inline]
    pub fn try_bucket_size(&mut self, bucket_size: usize) -> Result<&mut Self, InputError> {
        self.bucket_size =
            NonZeroUsize::new(bucket_size).ok_or(InputError::InvalidValueUsize(0, 0))?;
        Ok(self)
    }
    #[inline]
    pub fn split_method(&mut self, split_method: FindSplit) -> Result<&mut Self, InputError> {
        self.split_method = split_method;
        Ok(self)
    }
    #[inline]
    pub fn balancing(&mut self, balancing: &'a Matrix<'a>) -> Result<&mut Self, InputError> {
        InputError::check_sizes(balancing.nrow(), self.probabilities.len())?;
        self.balancing = Some(balancing);
        Ok(self)
    }
    #[inline]
    pub fn random_values(&mut self, random_values: &'a [f64]) -> Result<&mut Self, InputError> {
        InputError::check_sizes(random_values.len(), self.probabilities.len())?;
        self.random_values = Some(random_values);
        Ok(self)
    }
    #[inline]
    pub fn sample<R>(&self, rng: &mut R, sampler: Sampler<R>) -> Result<Vec<usize>, SamplingError>
    where
        R: Rng + ?Sized,
    {
        sampler(rng, self)
    }
    #[inline]
    pub fn check_spatially_balanced(&self) -> Result<&Self, InputError> {
        self.auxiliaries
            .ok_or_else(|| InputError::Missing("auxiliaries".to_owned()))?;

        Ok(self)
    }
    #[inline]
    pub fn check_balanced(&self) -> Result<&Self, InputError> {
        self.balancing
            .ok_or_else(|| InputError::Missing("balancing".to_owned()))?;

        Ok(self)
    }
    #[inline]
    pub fn check_coordinated(&self) -> Result<&Self, InputError> {
        self.random_values
            .ok_or_else(|| InputError::Missing("random_values".to_owned()))?;

        Ok(self)
    }
    #[inline]
    pub fn build_node(&self, units: &mut [usize]) -> Result<Box<Node<'a>>, SamplingError> {
        self.check_spatially_balanced()?;
        Ok(Box::new(
            TreeBuilder::new(self.auxiliaries.unwrap())
                .bucket_size(self.bucket_size)?
                .split_method(self.split_method)?
                .build(units)?,
        ))
    }
}

pub type Sampler<R> = fn(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>;
