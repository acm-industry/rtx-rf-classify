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

- `tensor.h`: Header file for the `TensorBase` class:
A compile-time-sized, mdspan-backed tensor type. Template parameters are `T` (floating-point type), `E` (fixed extents), and `A` (allocator, defaults to `std::allocator<T>`). All dimension sizes are encoded in the type via `std::extents`, enabling zero-overhead multi-dimensional access.
  - `TensorBase(allocator_type)`: Allocates and zero-initializes storage.
  - `TensorBase(const T* model_weights, allocator_type)`: Allocates and copies from a weight buffer.
  - `view()` -> `std::mdspan<T, E>&`: Returns the underlying mdspan view. Used to pass into `blas::gemm`, `blas::gemv`, and `blas::dot`.
  - `operator[](Idx... idx)` -> `T&`: Multi-dimensional element access.
  - `flat(size_t idx)` -> `T&`: Flat 1D element access over the contiguous storage.
  - `data()` -> `T*`: Raw pointer to the underlying memory.

- `blas_ops.h`: Type-safe, mdspan-aware BLAS wrappers with compile-time dimension checks.
Provides `gemm`, `gemv`, and `dot` operations that accept `std::mdspan` with fixed extents, template on the floating-point type (`float`/`double`), and dispatch to the appropriate CBLAS routine. Dimension compatibility is enforced via `static_assert`. All operations assume row-major layout (`CblasRowMajor`), matching `TensorBase`'s default `std::layout_right`.
  - `concept Matrix2D`: Requires a 2D mdspan with fixed extents and a floating-point element type.
  - `concept Vector1D`: Requires a 1D mdspan with fixed extents and a floating-point element type.
  - `void gemm(const MA& a, const MB& b, MC& c)`: Matrix-matrix multiply. A is M×K, B is K×N, C is M×N. Dispatches to `cblas_sgemm` (float) or `cblas_dgemm` (double).
  - `void gemv(const MA& a, const VX& x, VY& y)`: Matrix-vector multiply. A is M×N, x is N, y is M. Dispatches to `cblas_sgemv`/`cblas_dgemv`.
  - `T dot(const VA& a, const VB& b)`: Dot product. Both vectors are length N. Returns the scalar result. Dispatches to `cblas_sdot`/`cblas_ddot`.

- `matrix_eval.h`: Bridge between the Expression system (element-wise 1D eval) and BLAS matrix operations (2D).
Materializes lazy expressions into `TensorBase` storage via flat indexing, then calls `blas::gemm` on the results. Provides two overloads:
  - `void matrix_eval(expr_a, mat_a, expr_b, mat_b, out)`: Materializes both expressions into their respective tensors, then computes `out = mat_a * mat_b` via BLAS. Static asserts verify that expression sizes match tensor sizes.
  - `void matrix_eval(mat_a, mat_b, out)`: Directly calls `blas::gemm` when tensors are already materialized (no expression eval step).