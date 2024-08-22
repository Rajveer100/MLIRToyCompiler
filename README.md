# MiniToyCompiler

Mini Toy MLIR Compiler lowering a sequence of matrix operations, with Tiling and Bufferization.

# Description

The compiler supports basic operations for matrices including `ConstantOp`, `AddOp`, `MulOp` using
operation definition specification (ODS) via tablegen. Support for tiling and bufferization is also
provided via custom passes (more details below).

The operations are lowered to LLVM by validating the dialect and operations appropriately which
can be piped to the `mlir-cpu-runner`.

# Build Tools and Versioning

- CMake 3.29.6
- Ninja 1.12.1
- LLVM / MLIR 18.1.8 (HomeBrew Standalone)

# Usage

As per the current CMake config (can be updated as needed), to build:

(Project Root)

```Shell

cmake -G "Ninja" -S ./ -B build/
ninja -C build/

```

Once build is successful, the `mini-opt` tool will be generated. The tests can be executed manually
as of now, support for LLVM Lit infrastructure will be added soon.

```Shell

build/bin/mini-opt test/<test_folder>/<test_file>

```

Passes supported:

- `-tile-pass`
- `-one-shot-bufferize`
- `-dump-llvm-ir` (in progress)

Once the `-dump-llvm-ir` pass is complete, it can be piped to the MLIR cpu runner (will be added soon).
