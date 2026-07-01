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

//! Raw data -- internal data structure

/// Data container trait
pub trait RawData: Sized {
    type Elem;
    #[must_use]
    fn data(&self) -> &[Self::Elem];
}
/// Mutable data container trait
pub trait RawDataMut: RawData {
    #[must_use]
    fn data_mut(&mut self) -> &mut [Self::Elem];
}

impl<N> RawData for Vec<N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.as_slice() }
}
impl<N> RawDataMut for Vec<N> {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self.as_mut_slice() }
}
impl<N> RawData for &[N] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
