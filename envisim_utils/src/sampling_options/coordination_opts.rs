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

#[derive(Clone, Debug)]
pub struct CoordinationOptions<'a> {
    data: &'a [f64],
}

impl<'a> CoordinationOptions<'a> {
    #[inline]
    pub fn new(data: &'a [f64]) -> SamplingOptionsResult<Self> { Ok(Self { data }) }
    #[inline]
    pub fn data(&self) -> &'a [f64] { self.data }
}
