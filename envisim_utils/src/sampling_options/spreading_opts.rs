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

use std::num::NonZeroUsize;

use super::{
    SamplingOptionsError,
    SamplingOptionsResult,
};
use crate::kd_tree::split_methods::MidpointSlide;
use crate::kd_tree::{
    PointSet,
    Tree,
    TreeConfig,
};
use crate::number_traits::Number;

#[derive(Clone, Debug)]
pub struct SpreadingOptions<P> {
    data: P,
    bucket_size: NonZeroUsize,
}
impl<P> SpreadingOptions<P> {
    #[inline]
    pub fn data(&self) -> &P { &self.data }
    #[inline]
    pub fn bucket_size(&self) -> NonZeroUsize { self.bucket_size }
    #[inline]
    pub fn new<N>(data: P) -> SamplingOptionsResult<Self>
    where
        P: PointSet<N>,
    {
        Ok(Self {
            bucket_size: Self::estimate_bucket_size(data.size()),
            data,
        })
    }
    #[inline]
    pub fn set_bucket_size<NZ>(mut self, size: NZ) -> SamplingOptionsResult<Self>
    where
        NZ: TryInto<NonZeroUsize>,
    {
        self.bucket_size = size
            .try_into()
            .map_err(|_| SamplingOptionsError::InvalidBucketSize)?;
        Ok(self)
    }
    #[inline]
    fn estimate_bucket_size(n_units: NonZeroUsize) -> NonZeroUsize {
        let bucket_size = match n_units.get() {
            0..=100 => 10,
            101..=400 => n_units.get() / 10,
            _ => 40,
        };
        NonZeroUsize::new(bucket_size).unwrap()
    }
    #[inline]
    pub fn to_tree<N>(&self) -> Tree<'_, N, P>
    where
        P: PointSet<N>,
        N: Number,
    {
        let mut units: Vec<usize> = self.data.id_iter().collect();
        Tree::new(self, &mut units)
    }
}
impl<P, N> TreeConfig<N> for SpreadingOptions<P>
where
    P: PointSet<N>,
    N: Number,
{
    type Data = P;
    type Split = MidpointSlide<N>;
    #[inline]
    fn data(&self) -> &Self::Data { &self.data }
    #[inline]
    fn bucket_size(&self) -> NonZeroUsize { self.bucket_size }
    #[inline]
    fn split_method(&self, units: &[usize]) -> Self::Split { MidpointSlide::new(&self.data, units) }
}
