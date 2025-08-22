// Copyright (C) 2025 Wilmer Prentius.
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

use crate::SamplingError;
use envisim_utils::kd_tree::{midpoint_slide, FindSplit, Node, TreeBuilder};
use envisim_utils::{random::RandomNumberGenerator, InputError, Matrix, Probabilities};
use std::num::NonZeroUsize;

pub struct AuxiliariesOptions<'a> {
    data: &'a Matrix<'a>,
    bucket_size: NonZeroUsize,
    split_method: FindSplit,
}

impl<'a> AuxiliariesOptions<'a> {
    #[inline]
    pub fn new(data: &'a Matrix<'a>) -> Result<Self, InputError> {
        let options = AuxiliariesOptions {
            data,
            bucket_size: unsafe { NonZeroUsize::new_unchecked(40) },
            split_method: midpoint_slide,
        }
        .est_bucket_size()?;
        Ok(options)
    }

    #[inline]
    pub fn check(&self, size: usize) -> Result<&Self, InputError> {
        InputError::check_sizes(self.data.nrow(), size)?;
        Ok(self)
    }

    #[inline]
    pub fn data(&self) -> &'a Matrix<'a> {
        self.data
    }
    #[inline]
    pub fn bucket_size(&self) -> NonZeroUsize {
        self.bucket_size
    }
    #[inline]
    pub fn split_method(&self) -> FindSplit {
        self.split_method
    }

    #[inline]
    pub fn set_data(mut self, data: &'a Matrix<'a>) -> Result<Self, InputError> {
        self.data = data;
        Ok(self)
    }
    #[inline]
    pub fn set_bucket_size(mut self, bucket_size: NonZeroUsize) -> Result<Self, InputError> {
        self.bucket_size = bucket_size;
        Ok(self)
    }
    #[inline]
    pub fn try_bucket_size(self, bucket_size: usize) -> Result<Self, InputError> {
        self.set_bucket_size(
            NonZeroUsize::new(bucket_size).ok_or(InputError::InvalidValueUsize(0, 0))?,
        )
    }
    #[inline]
    pub fn est_bucket_size(self) -> Result<Self, InputError> {
        let len = self.data.nrow();
        self.set_bucket_size(unsafe {
            NonZeroUsize::new_unchecked(match len {
                0usize..=100 => 10usize,
                101usize..=400 => len / 10usize,
                _ => 40usize,
            })
        })
    }
    #[inline]
    pub fn set_split_method(mut self, split_method: FindSplit) -> Result<Self, InputError> {
        self.split_method = split_method;
        Ok(self)
    }

    #[inline]
    pub fn build_tree(&self, units: &mut [usize]) -> Result<Box<Node<'a>>, SamplingError> {
        Ok(Box::new(
            TreeBuilder::new(self.data)
                .bucket_size(self.bucket_size)?
                .split_method(self.split_method)?
                .build(units)?,
        ))
    }
}

pub struct SampleOptions<'a> {
    // Base
    probabilities: &'a [f64],
    eps: f64,
    max_iterations: NonZeroUsize,

    // Coordinated
    random_values: Option<&'a [f64]>,

    // Spatially balanced sampling
    spreading: Option<AuxiliariesOptions<'a>>,

    // Balanced sampling
    balancing: Option<&'a Matrix<'a>>,
}

impl<'a> SampleOptions<'a> {
    #[inline]
    pub fn new(probabilities: &'a [f64]) -> Result<Self, InputError> {
        Probabilities::check(probabilities)?;
        Ok(Self {
            probabilities,
            eps: 1e-12,
            max_iterations: unsafe { NonZeroUsize::new_unchecked(1000) },
            random_values: None,
            spreading: None,
            balancing: None,
        })
    }
    #[inline]
    pub fn check_base(&self) -> Result<&Self, InputError> {
        Probabilities::check_eps(self.eps)?;
        Ok(self)
    }
    #[inline]
    pub fn check_spreading(&self) -> Result<&Self, InputError> {
        let len = self.population_size();

        if let Some(spr) = self.spreading.as_ref() {
            spr.check(len)?;
        } else {
            return Err(InputError::IsNone);
        }

        Ok(self)
    }
    #[inline]
    pub fn check_balancing(&self) -> Result<&Self, InputError> {
        let len = self.population_size();

        if let Some(bal) = self.balancing {
            InputError::check_sizes(bal.nrow(), len)?;
        } else {
            return Err(InputError::IsNone);
        }

        Ok(self)
    }

    #[inline]
    pub fn population_size(&self) -> usize {
        self.probabilities.len()
    }
    #[inline]
    pub fn probabilities(&self) -> &'a [f64] {
        self.probabilities
    }
    #[inline]
    pub fn eps(&self) -> f64 {
        self.eps
    }
    #[inline]
    pub fn max_iterations(&self) -> NonZeroUsize {
        self.max_iterations
    }
    #[inline]
    pub fn random_values(&self) -> Option<&'a [f64]> {
        self.random_values
    }

    #[inline]
    pub fn set_probabilities(mut self, probabilities: &'a [f64]) -> Result<Self, InputError> {
        InputError::check_lengths(probabilities, self.probabilities)?;
        Probabilities::check(probabilities)?;
        self.probabilities = probabilities;
        Ok(self)
    }
    #[inline]
    pub fn set_eps(mut self, eps: f64) -> Result<Self, InputError> {
        self.eps = Probabilities::check_eps(eps)?;
        Ok(self)
    }
    #[inline]
    pub fn set_max_iterations(mut self, max_iterations: NonZeroUsize) -> Result<Self, InputError> {
        self.max_iterations = max_iterations;
        Ok(self)
    }
    #[inline]
    pub fn set_random_values(mut self, random_values: &'a [f64]) -> Result<Self, InputError> {
        InputError::check_sizes(random_values.len(), self.population_size())?;
        self.random_values = Some(random_values);
        Ok(self)
    }

    #[inline]
    pub fn spreading(&self) -> Option<&AuxiliariesOptions<'a>> {
        self.spreading.as_ref()
    }
    pub fn set_spreading(mut self, spreading: &'a Matrix<'a>) -> Result<Self, InputError> {
        self.spreading = Some(AuxiliariesOptions::new(spreading)?);
        Ok(self)
    }
    #[inline]
    pub fn set_spreading_options(
        mut self,
        spreading: AuxiliariesOptions<'a>,
    ) -> Result<Self, InputError> {
        self.spreading = Some(spreading);
        Ok(self)
    }
    #[inline]
    pub fn remove_spreading(mut self) -> Result<Self, InputError> {
        self.spreading = None;
        Ok(self)
    }

    #[inline]
    pub fn balancing(&self) -> Option<&'a Matrix<'a>> {
        self.balancing
    }
    #[inline]
    pub fn set_balancing(mut self, balancing: &'a Matrix<'a>) -> Result<Self, InputError> {
        self.balancing = Some(balancing);
        Ok(self)
    }
    #[inline]
    pub fn remove_balancing(mut self) -> Result<Self, InputError> {
        self.balancing = None;
        Ok(self)
    }

    #[inline]
    pub fn sample<R>(&self, rng: &mut R, sampler: Sampler<R>) -> Result<Vec<usize>, SamplingError>
    where
        R: RandomNumberGenerator + ?Sized,
    {
        sampler(rng, self)
    }
}

pub type Sampler<R> = fn(rng: &mut R, options: &SampleOptions) -> Result<Vec<usize>, SamplingError>;
