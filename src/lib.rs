//! neural-plato Rust FFI bindings
//!
//! Safe Rust wrappers around the Fortran computational primitives.

mod sparse_memory;
mod dream_backend;
pub mod penrose_palace;
pub mod penrose_bit;
pub mod penrose_floor;
pub mod cut_and_project;
pub mod nasty_compress;

pub use sparse_memory::SparseMemoryLayer;
pub use dream_backend::{DreamConfig, DreamResult};
pub use penrose_palace::{PenrosePalace, PenroseTile, TileType, ShardColor, RetrievedMemory};
pub use penrose_bit::{PenroseBrain, PenroseAddress, PenroseBit};
pub use penrose_floor::{PenroseFloor, Step, FloorRead};
pub use cut_and_project::CutAndProject;
pub use nasty_compress::{NastyCompress, CompressedTile, Decompressed};

/// Library version, matching the Fortran module.
pub const VERSION: &str = "0.1.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
