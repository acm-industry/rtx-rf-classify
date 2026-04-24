# fftw_ops.h Documentation

## Description
This file provides a type-safe object-oriented wrapper around the FFTW library. It bridges the gap between raw `fftw_complex` C arrays and `TensorBase`, bringing compile-time optimizations and expression-system integrations to Fourier transforms.

## Classes

### `template <typename T> struct fftw_traits;`
A type trait mapping `float`, `double`, and `std::complex` variations to their underlying FFTW functions (`fftw_plan_dft`, `fftwf_execute`, etc.) and data types, avoiding duplicate implementation loops.

### `template <typename T, FixedExtent E, int Sign = FFTW_FORWARD, unsigned Flags = FFTW_ESTIMATE> class FFTW;`
The main Complex-to-Complex Fourier Transform handler. 

- `T`: The underlying floating point type or complex type (`float`, `double`, `std::complex<float>`, `std::complex<double>`).
- `E`: Complete `std::extents` specification defining the shape of the transformation. Arbitrary rank is supported; the extents are materialized into a stack `std::array<int, E::rank()>` at plan-creation time and forwarded to `fftw_plan_dft(rank, n, ...)`.
- `Sign`: Represents the transform direction (`FFTW_FORWARD`, `FFTW_BACKWARD`).
- `Flags`: Defines FFTW optimization efforts for this particular plan (e.g., `FFTW_MEASURE` vs `FFTW_ESTIMATE`).

### `template <typename T, FixedExtent E, unsigned Flags = FFTW_ESTIMATE> class FFTW_R2C;`
The Real-to-Complex Fourier Transform handler. Converts a real layout tensor into an optimally smaller complex layout representing mirrored frequency domain buckets. Currently rank-1 only (uses the FFTW 1D entry point `fftw_plan_dft_r2c_1d`).

### `template <typename T, FixedExtent OutE, unsigned Flags = FFTW_ESTIMATE> class FFTW_C2R;`
The Complex-to-Real Fourier Transform handler. Restores a real layout tensor from a half-length frequency format mirrored structure. Currently rank-1 only (uses the FFTW 1D entry point `fftw_plan_dft_c2r_1d`).

#### Constructor
- `FFTW() = default;` (Same for `FFTW_R2C` and `FFTW_C2R`)
  Instances are generated empty. The internal `plan_` will only be concretized via an internal `create_plan()` internally upon its first invocation against actual memory blocks. Copying is disabled, but move construction/assignment is permitted. 

#### Methods

- `void operator()(TensorBase<in_t, E>& in, TensorBase<out_t, OutExtent>& out);`
  The main interaction point. Performs the FFT and stores it in `out`. At compile-time, this checks that the `in` and `out` arrays match exactly the memory dimensions required by the transformation direction. Using `TensorBase` references allows seamless generic mapping directly from `DynTensor` or externally managed raw buffers alike.

- `template <Expression Expr> void eval(const Expr& expr, TensorBase<in_t, E>& temp_in, TensorBase<out_t, OutExtent>& out);`
  Materializes any generic math `Expression` into the `temp_in` sequence using iterator bindings, then automatically invokes `operator()` to produce `out`. The shape validation relies on compile-time checks making sure `expr::iter_size() == compute_static_size<E>()`.
