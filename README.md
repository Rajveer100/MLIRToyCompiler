# MLIRToyCompiler

MLIR Toy Compiler lowering a sequence of matrix operations, with Tiling and Bufferization.

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
as of now, support for LLVM Lit infrastructure and FileCheck will be added soon.

Passes supported:

- `-tile-ops`
- `-one-shot-bufferize`
- `-lower-to-affine`
- `-lower-to-llvm`

Examples:

Tiling:

```Shell
build/bin/mini-opt -tile-ops test/<test_folder>/<test_file>
```

Tiling with Bufferization:

```Shell
build/bin/mini-opt -tile-ops -one-shot-bufferize test/<test_folder>/<test_file>
```

Affine (partial) lowering:

```Shell
build/bin/mini-opt -lower-to-affine test/<test_folder>/<test_file>
```

Affine (partial) + LLVM (full) lowering:

```Shell
build/bin/mini-opt -lower-to-affine -lower-to-llvm test/<test_folder>/<test_file>
```

Note:

The following pass combination requires `ExtractSliceOpLowering` and `InsertSliceOpLowering`
which will be added in the next commit:

```Shell
build/bin/mini-opt -tile-ops -lower-to-affine -lower-to-llvm test/<test_folder>/<test_file>
```

The full lowering can then be piped to the `mlir-translate` tool 
to get final LLVM IR:

```Shell
build/bin/mini-opt -lower-to-affine -lower-to-llvm test/<test_folder>/<test_file> | 
/opt/homebrew/Cellar/llvm/18.1.8/bin/mlir-translate -mlir-to-llvmir
```

This can further be passed to `llc` to get assembly code:

```Shell
build/bin/mini-opt -lower-to-affine -lower-to-llvm test/<test_folder>/<test_file> | 
/opt/homebrew/Cellar/llvm/18.1.8/bin/mlir-translate -mlir-to-llvmir |
/opt/homebrew/Cellar/llvm/18.1.8/bin/llc
```
