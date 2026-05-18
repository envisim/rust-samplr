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

#[expect(
    clippy::field_scoped_visibility_modifiers,
    reason = "ok within matrix module"
)]
#[must_use]
#[derive(Debug, Clone)]
pub struct OwnedMatrixData<N> {
    /// Internal data vector
    pub(super) data: Vec<N>,
}
impl<N> OwnedMatrixData<N> {
    /// Constructs a new owned matrix data
    #[inline]
    pub fn new(data: Vec<N>) -> Self { Self { data } }
}

impl<N> RawData for OwnedMatrixData<N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { &self.data }
}
impl<N> From<&BorrowedMatrixData<'_, N>> for OwnedMatrixData<N>
where
    N: Copy,
{
    #[inline]
    fn from(data: &BorrowedMatrixData<N>) -> Self { Self::new(data.data().to_vec()) }
}
impl<N> From<&[N]> for OwnedMatrixData<N>
where
    N: Copy,
{
    #[inline]
    fn from(data: &[N]) -> Self { Self::new(data.to_vec()) }
}
impl<N> From<Vec<N>> for OwnedMatrixData<N> {
    #[inline]
    fn from(data: Vec<N>) -> Self { Self::new(data) }
}

#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct BorrowedMatrixData<'bdata, N> {
    /// Internal data reference
    data: &'bdata [N],
}
impl<'bdata, N> BorrowedMatrixData<'bdata, N> {
    /// Constructs a new borrowed matrix data
    #[inline]
    pub fn new(data: &'bdata [N]) -> Self { Self { data } }
}

impl<N> RawData for BorrowedMatrixData<'_, N> {
    type Elem = N;
    #[inline]
    fn data(&self) -> &[Self::Elem] { self.data }
}
impl<'borrow, N> From<&'borrow OwnedMatrixData<N>> for BorrowedMatrixData<'borrow, N> {
    #[inline]
    fn from(data: &'borrow OwnedMatrixData<N>) -> Self { Self::new(data.data()) }
}
impl<'bdata, N> From<&'bdata [N]> for BorrowedMatrixData<'bdata, N> {
    #[inline]
    fn from(data: &'bdata [N]) -> Self { Self::new(data) }
}
