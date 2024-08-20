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

//! Implementation of a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree), together with search
//! capabilities needed by the sampling and estimations methods.
//!
//! # References
//! Lisic, J. J., & Cruze, N. B. (2016).
//! Local pivotal methods for large surveys.
//! In proceedings, ICES V, Geneva Switzerland 2016.
//! In Proceedings of the Fifth International Conference on Establishment Surveys.

mod node;
mod searcher;
mod split_methods;

pub use node::*;
pub use searcher::*;
pub use split_methods::midpoint_slide;
