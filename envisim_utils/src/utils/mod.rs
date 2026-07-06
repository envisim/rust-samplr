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

//! Small utility functions

mod epsilon;
mod number_traits;
mod slice_view;
mod spatial;
pub mod summary_statistics;

pub use epsilon::Epsilon;
pub use number_traits::{
    Number,
    NumberFloat,
    NumberInt,
};
pub use slice_view::{
    SliceView,
    SliceViewMut,
};
pub use spatial::PointSet;
