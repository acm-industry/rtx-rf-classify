# Source Code

## Purpose

This directory contains the C++ source code for the Systems project. The code is designed to be compilable for both native architectures and RISC-V via cross-compilation.

## RISC-V Vectorization Pipeline

The source code in this directory is compiled for RISC-V architecture as part of the vectorization verification pipeline. The pipeline:

1. **Cross-compiles** the source code using Clang with `-march=rv64gcv` targeting the RISC-V Vector extension
2. **Verifies vectorization** by checking that the compiler emits RVV instructions in the assembly output
3. **Tests correctness** by running the compiled RISC-V binary on Spike (the RISC-V ISA simulator)

The test kernels in `tests/test_rvv_kernels.cpp` mirror the project's real operations (vector add, dot product, matrix-vector multiply, fused multiply-add, conv1d, relu, maxpool, linear) and are used to validate that each operation is vectorized and produces correct results.

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
- `Allocator<T> get_allocator<T, size_t>()`: Returns an instance of `Allocator<T, size_t>`. Most if not all interactions with the buffer should go through an Allocator. The `size_t` parameter dictates alignment size, allowing for optimal SIMD loads.
- `class Allocator<T, size_t>`: This is a STL-container compatible allocator for obtaining memory from the `MemoryBuffer`. It follows the named requirements for an Allocator.


- `memorybuffer.cpp`: Some impls for some of the files in `memorybuffer.h`.

- `tensor.h`: Header file for the `TensorBase` class:
A TensorBase class which is effectively a custom view onto a pointer. It allows for several niceties:
  - View semantics: TensorBase does not assume ownership of it's own data. Therefore, we can do many things recursively. For example, we
  `TensorBase<T, std::extents<size_t, 4, 4>>(data)[0]` will return a `TensorBase<T, std::extents<size_t, 4>`. You can also use `flat_view`
  to get a `TensorBase` with rank 1, but mainting the same size (akin to numpy `flatten`). This also extends to the iterator type; for example,
  the same 4x4 tensor above will return a rank-1, length 4 tensor when iterating over with a for-each loop. This is optimized too; on a rank-N tensor,
  an N for loop should (ideally) have similar or exactly the same performance as direct loops (`(i = 0; i < ...)`).
  - Lazy evaluation compatibility: TensorBase is automatically compatible with the expression system. 
  - Compile time shape assurance; the Tensor will always ensure rank > 0 and that its total size is preserved under operations that require it
  like a move or copy between Tensors of different shapes. 
Also provides `DynTensor`, which is a subclass of TensorBase that manages its own memory. An allocator is usable to defer allocation to a `MemoryBuffer`.

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
