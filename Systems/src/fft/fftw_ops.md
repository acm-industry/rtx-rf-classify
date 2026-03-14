# fftw_ops.h Documentation

## Description
This file provides a type-safe object-oriented wrapper around the FFTW library. It bridges the gap between raw `fftw_complex` C arrays and `TensorBase`, bringing compile-time optimizations and expression-system integrations to Fourier transforms.

## Classes

### `template <typename T> struct fftw_traits;`
A type trait mapping `float`, `double`, and `std::complex` variations to their underlying FFTW functions (`fftw_plan_dft`, `fftwf_execute`, etc.) and data types, avoiding duplicate implementation loops.

### `template <typename T, FixedExtent E, int Sign = FFTW_FORWARD, unsigned Flags = FFTW_ESTIMATE> class FFTW;`

The main Fourier Transform handler. 

- `T`: The underlying floating point type or complex type (`float`, `double`, `std::complex<float>`, `std::complex<double>`).
- `E`: Complete `std::extents` specification defining the shape of the transformation. Only single-rank transforms are currently supported. 
- `Sign`: Represents the transform direction (`FFTW_FORWARD`, `FFTW_BACKWARD`).
- `Flags`: Defines FFTW optimization efforts for this particular plan (e.g., `FFTW_MEASURE` vs `FFTW_ESTIMATE`).

#### Constructor
- `FFTW() = default;`
  Instances are generated empty. The internal `plan_` will only be concretized via `create_plan()` internally upon its first invocation against actual memory blocks. Copying is disabled, but move construction/assignment is permitted. 

#### Methods

- `template <Allocator<val_t> AllocIn, Allocator<val_t> AllocOut> void operator()(DynTensor<val_t, E, AllocIn>& in, DynTensor<val_t, E, AllocOut>& out);`
  The main interaction point. Performs the FFT and stores it in `out`. At compile-time, this checks that the `in` and `out` arrays exactly match the dimensions expected by `E`.

- `template <Expression Expr, Allocator<val_t> AllocIn, Allocator<val_t> AllocOut> void eval(const Expr& expr, DynTensor<val_t, E, AllocIn>& temp_in, DynTensor<val_t, E, AllocOut>& out);`
  Materializes any generic math `Expression` into the `temp_in` sequence using iterator bindings, then automatically invokes `operator()` to produce `out`. The shape validation relies on compile-time checks making sure `expr::iter_size() == compute_static_size<E>()`.
