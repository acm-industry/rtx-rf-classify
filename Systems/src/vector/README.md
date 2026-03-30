# Vector Operations

## Purpose

This directory provides a portable vector/matrix operations interface with multiple compile-time-selectable backends. The public API (`vector_ops.h`) exposes:
- binary/linear ops: element-wise addition, dot product, matrix-matrix multiplication, and matrix-vector multiplication
- unary ops: `relu`, `exp`, `log`, `tanh`, `sigmoid`
All operations work on raw `float*` pointers with explicit sizes.

For type-safe, mdspan-based wrappers with compile-time dimension checks, see `blas_ops.h` in the parent directory.

## Backend Selection

The backend is chosen at compile time via preprocessor defines, in the following priority order:

| Priority | Flag | Backend | File | Description |
|----------|------|---------|------|-------------|
| 1 | `-DUSE_BLAS` | OpenBLAS | `impl/vector_blas.cpp` | Delegates `matmul`, `gemv`, and `dot` to CBLAS routines. `add` uses a sequential fallback since BLAS `saxpy` is in-place and doesn't match the out-of-place API. |
| 2 | `-DUSE_SLEEF` | SLEEF math backend | `impl/vector_sleef.cpp` | Uses SLEEF scalar entry points for `exp`, `log`, `tanh` (and `sigmoid` via `exp`). If `<sleef.h>` is unavailable, transparently falls back to `std::*`. Linear algebra ops currently use scalar loops in this backend. |
| 3 | `-DUSE_HWY` (or `-DUSE_HIGHWAY`) | Highway SIMD | `impl/vector_hwy.cpp` | Uses [Google Highway](https://github.com/google/highway) for portable SIMD across x86 (SSE/AVX) and RISC-V. Handles tail elements via `LoadN`/`StoreN`. |
| 4 | (none) | Sequential | `impl/vector_seq.cpp` | Plain scalar loops. Portable across all architectures. |

The dispatcher (`vector_ops.cpp`) includes exactly one backend via:
`#if defined(USE_BLAS) ... #elif defined(USE_SLEEF) ... #elif defined(USE_HWY)||defined(USE_HIGHWAY) ... #else ...`
and forwards public API calls to the corresponding `impl_*` functions.

## Files

- `vector_ops.h`: Public interface.
  - Linear ops: `vec::add`, `vec::dot`, `vec::matmul`, `vec::matvec`
  - Unary ops: `vec::relu`, `vec::exp`, `vec::log`, `vec::tanh`, `vec::sigmoid`
  - `void add(const float* a, const float* b, float* out, size_t n)`: Element-wise addition. `out[i] = a[i] + b[i]`.
  - `void dot(const float* a, const float* b, float* out, size_t n)`: Dot product. `*out = sum(a[i] * b[i])`.
  - `void matmul(const float* a, const float* b, float* out, size_t rows_a, size_t cols_a, size_t cols_b)`: Row-major matrix multiply. `out = A * B`.
  - `void matvec(const float* a, const float* x, float* out, size_t rows, size_t cols)`: Row-major matrix-vector multiply. `out = A * x`.
  - `void relu(const float* in, float* out, size_t n)`: Element-wise ReLU. `out[i] = max(0, in[i])`.
  - `void exp(const float* in, float* out, size_t n)`: Element-wise exponential.
  - `void log(const float* in, float* out, size_t n)`: Element-wise natural logarithm.
  - `void tanh(const float* in, float* out, size_t n)`: Element-wise hyperbolic tangent.
  - `void sigmoid(const float* in, float* out, size_t n)`: Element-wise logistic sigmoid.

- `vector_ops.cpp`: Backend dispatcher. Includes one impl file based on compile-time defines, then wraps `impl_*` calls in the `vec::` namespace.

- `impl/vector_seq.cpp`: Sequential (scalar) backend. Simple `for` loops, no dependencies.

- `impl/vector_hwy.cpp`: Highway SIMD backend. Uses `ScalableTag<float>` for lane detection, `LoadN`/`StoreN` for tail handling, `MulAdd` for fused multiply-add, and `ReduceSum` for reductions. Unary transcendental ops currently use scalar `std::*` loops here, with SIMD ReLU.

- `impl/vector_blas.cpp`: OpenBLAS backend. Calls `cblas_sdot`, `cblas_sgemm`, `cblas_sgemv` for dot, matmul, and matvec respectively. Falls back to a sequential loop for add.

- `impl/vector_sleef.cpp`: SLEEF backend. Uses SLEEF math for `exp`, `log`, `tanh` (and `sigmoid` through `exp`) when `<sleef.h>` is available. Includes guarded fallback to `std::*` if SLEEF headers are not present.

- `tests/test_vector_ops.cpp`: 14 tests covering all four operations across basic, edge-case, and large-size scenarios. Includes timing output.

- `tests/test_blas_ops.cpp`: 8 tests for the mdspan BLAS wrappers in `blas_ops.h` (gemm, gemv, dot for float and double).

- `tests/test_matrix_eval.cpp`: Tests for `matrix_eval.h`, verifying expression materialization into tensors followed by BLAS gemm, including expression operations (negation, scaling) and the direct tensor overload.
