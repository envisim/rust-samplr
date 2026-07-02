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

//! Internal data structure traits

use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

/// Data container trait
pub trait RawData: Sized {
    type Elem;
    /// Returns a reference (view) to the internal data, expected to be in column major order.
    #[must_use]
    fn data(&self) -> &[Self::Elem];
}
/// Mutable data container trait
pub trait RawDataMut: RawData {
    /// Returns a mutable reference to the internal data, expected to be in column major order.
    #[must_use]
    fn data_mut(&mut self) -> &mut [Self::Elem];
}

// View containers
impl<N, const L: usize> RawData for [N; L] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for &[N] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for &mut [N] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for Vec<N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for Box<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for Cow<'_, [N]>
where
    N: Clone,
{
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for Rc<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> RawData for Arc<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}

// Mutable containers
impl<N, const L: usize> RawDataMut for [N; L] {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> RawDataMut for &mut [N] {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> RawDataMut for Vec<N> {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> RawDataMut for Box<[N]> {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> RawDataMut for Cow<'_, [N]>
where
    N: Clone,
{
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self.to_mut() }
}
