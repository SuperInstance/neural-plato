//! neural-plato Rust FFI bindings
//!
//! Safe Rust wrappers around the Fortran computational primitives.

mod sparse_memory;
mod dream_backend;

pub use sparse_memory::SparseMemoryLayer;
pub use dream_backend::{DreamConfig, DreamResult};

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
