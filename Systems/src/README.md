# Source Code

## Purpose

This directory contains the C++ source code for the Systems project. The code is designed to be compilable for both native architectures and RISC-V via cross-compilation.

## RISC-V CI Pipeline Integration

The source code in this directory is compiled for RISC-V architecture as part of the CI pipeline validation process. The pipeline:

1. **Cross-compiles** the source code using the RISC-V toolchain configured in `cmake/toolchains/riscv64-linux-gnu.cmake`
2. **Links statically** (on non-MSVC platforms) to ensure the binary is self-contained
3. **Tests execution** by running the compiled RISC-V binary using QEMU emulation

This ensures that the codebase is compatible with RISC-V architecture and can be successfully built and executed in RISC-V environments. The CI pipeline validates that:
- The code compiles without errors for RISC-V
- The binary executes correctly when emulated
- No architecture-specific assumptions are made in the code

## Files
The `test` folder contains tests for some of the applications below. These should go unused unless explicitly for testing.

- `main.cpp`: Main entry point for the application

- `memorybuffer.h`: Header file for MemoryBuffer class:
Contains a `MemoryBuffer` class for bump-allocating memory from a dedicated memory pool.<br>
`MemoryBuffer` has two constructors:
1. `MemoryBuffer(size_t)`: The memory buffer will heap-allocate and manage its own memory. It will allocate the size passed in (in bytes).
2. `MemoryBuffer(std::span<std::byte>)`: The memory buffer will not assume ownership of the memory, but will instead just manage it. This should primarily be used if static or stack memory is usable. Heap memory should use constructor 1. <br>
**Note**: The copy and move constructor/assignment methods have all been deleted. This is to ensure that the MemoryBuffer instance is bounded to the scope it is created in. This helps avoid unintended moving of large amounts of data. 
The class also exposes: <br>
- `Allocator<T> get_allocator<T>()`: Returns an instance of `Allocator<T>`. Most if not all interactions with the buffer should go through an Allocator.
- `class Allocator<T>`: This is a STL-container compatible allocator for obtaining memory from the `MemoryBuffer`. It follows the named requirements for an Allocator.


- `memorybuffer.cpp`: Some impls for some of the files in `memorybuffer.h`.