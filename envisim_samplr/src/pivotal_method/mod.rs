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

mod base;
mod hierarchical;
mod runner;
mod spatial;

pub use base::{
    PivotalSampling,
    RandomStrategy,
    SequentialStrategy,
};
pub use hierarchical::hierarchical_lpm_2;
pub use runner::{
    PivotalRunner,
    PivotalStrategy,
};
pub use spatial::{
    LocalPivotalSampling,
    LocalStrategy1,
    LocalStrategy1S,
    LocalStrategy2,
};
