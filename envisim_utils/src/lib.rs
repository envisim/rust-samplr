// Copyright (C) 2024 Wilmer Prentius, Anton Grafström.
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

//! Utility functions for envisim

mod error;
mod indices;
pub mod kd_tree;
mod matrix;
pub mod pips;
mod probabilities;
pub mod utils;

pub use error::InputError;
pub use indices::{Indices, IndicesError};
pub use matrix::Matrix;
pub use probabilities::Probabilities;
