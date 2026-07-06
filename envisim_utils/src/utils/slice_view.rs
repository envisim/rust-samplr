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

//! Views for sliceables
//!
//! Mirroring `AsRef` and `AsMut`, but using associated type.

use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

/// Data container trait
pub trait SliceView {
    type Elem;
    /// Returns a reference (view) to the internal data.
    #[must_use]
    fn data(&self) -> &[Self::Elem];
}
/// Mutable data container trait
pub trait SliceViewMut: SliceView {
    /// Returns a mutable reference to the internal data.
    #[must_use]
    fn data_mut(&mut self) -> &mut [Self::Elem];
}

// View containers
impl<N, const L: usize> SliceView for [N; L] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for &[N] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for &mut [N] {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for Vec<N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for Box<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for Cow<'_, [N]>
where
    N: Clone,
{
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for Rc<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}
impl<N> SliceView for Arc<[N]> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self }
}

// Mutable containers
impl<N, const L: usize> SliceViewMut for [N; L] {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> SliceViewMut for &mut [N] {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> SliceViewMut for Vec<N> {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> SliceViewMut for Box<[N]> {
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self }
}
impl<N> SliceViewMut for Cow<'_, [N]>
where
    N: Clone,
{
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::Elem] { self.to_mut() }
}
