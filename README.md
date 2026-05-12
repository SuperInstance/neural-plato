# neural-plato ⚒️

Fortran + Rust hybrid implementing the core computational primitives behind sparse memory models, constraint theory, and experimental forgetting curves.

## Architecture

```
neural-plato/
├── fortran/          — Raw computational kernels (Fortran 2008)
│   ├── sparse_memory.f90    UltraMem-inspired sparse memory layer
│   ├── amnesia_curve.f90    Forgetting curve computation
│   ├── intent_snap.f90      Eisenstein lattice snap (vectorized)
│   ├── tucker_decompose.f90 TDQKR-style value retrieval
│   ├── negative_space.f90   Shadow reconstruction
│   └── neural_plato.f90     Main module re-exports
├── src/              — Rust FFI bindings + safe wrappers
│   ├── lib.rs               FFI declarations
│   ├── sparse_memory.rs     Safe Rust wrapper
│   └── dream_backend.rs     Dream config + types
└── examples/
    └── basic.f90            Fortran usage example
```

## Fortran Modules

### sparse_memory
UltraMem-style sparse memory: virtual table of value vectors with row/column routing, top-k selection, and implicit value expansion (4x virtual without memory cost).

### amnesia_curve
Experimental forgetting curves from baton protocol experiments. Predicts accuracy degradation as coverage drops, with style-dependent factors.

### intent_snap
Vectorized Eisenstein lattice snap. Snaps 2D positions to the nearest Eisenstein integer in the dodecet system (12-fold symmetry).

### tucker_decompose
Tucker Decomposed Query-Key Retrieval (TDQKR) from the UltraMem paper. Computes query-key scores via Tucker decomposition for efficient sparse memory lookup.

### negative_space
Shadow reconstruction from negative constraints. Computes the "shadow" — what can be inferred from what is explicitly absent.

## Building

```bash
# Build Fortran objects
cd fortran && make && cd ..

# Build Rust crate (links Fortran)
cargo build
```

## Requirements

- gfortran (Fortran 2008 compatible)
- rustc 1.75.0+
- No edition 2024 features

## License

MIT
