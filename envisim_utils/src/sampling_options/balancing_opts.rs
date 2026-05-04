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

use super::SamplingOptionsResult;
use crate::matrix::MatrixRef;

#[derive(Clone, Debug)]
pub struct BalancingOptions<'a, N> {
    data: MatrixRef<'a, N>,
}
impl<'a, N> BalancingOptions<'a, N> {
    #[inline]
    pub fn new(data: MatrixRef<'a, N>) -> SamplingOptionsResult<Self> { Ok(Self { data }) }
    #[inline]
    pub fn data(&self) -> &MatrixRef<'a, N> { &self.data }
}
