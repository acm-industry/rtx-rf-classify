# FFTW Wrapper

This module provides a C++ wrapper around FFTW3 for FFT-based signal preprocessing in the RF classification pipeline.

## Installation

FFTW3 is required for native builds:

```bash
# macOS (Homebrew)
brew install fftw

# Ubuntu/Debian
sudo apt install libfftw3-dev

# Fedora/RHEL
sudo dnf install fftw-devel
```

## Build

```bash
cd Systems
cmake -S . -B build/native -DUSE_FFTW=ON
cmake --build build/native --target test_fft
./build/native/test_fft
```

## FFTW Efficiency Guide

FFTW uses a **plan → execute** model that separates algorithm selection from computation. Understanding this model is critical for optimal performance.

### The Plan-Execute Model

```cpp
#include "fft/fftw_wrapper.h"

// 1. Allocate arrays (use fftw_malloc for SIMD alignment)
auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

// 2. Create plan ONCE (potentially expensive, depending on flag used)
auto plan = fft::PlanFactory::create_1d_c2c(N, in, out, FFTW_FORWARD, FFTW_MEASURE);

// 3. Execute MANY times
for (int i = 0; i < num_signals; ++i) {
    load_signal_data(in, i);
    plan.execute();           // Uses pre-computed optimal algorithm
    process_fft_result(out);
}

// 4. Cleanup
fftw_free(in);
fftw_free(out);
// plan automatically destroyed via RAII
```

### Sign Flags
`FFTW_FORWARD` is a forward pass of fft
`FFTW_BACKWARD` is for an inverse pass of fft

### Planning Flags

| Flag | Planning Time | Execution Speed | Use Case |
|------|---------------|-----------------|----------|
| `FFTW_ESTIMATE` | ~0ms | Good | Development, testing, one-off FFTs |
| `FFTW_MEASURE` | ~100ms-1s | Better | Production with repeated FFTs |
| `FFTW_PATIENT` | ~1-10s | Best | Batch processing, known sizes |
| `FFTW_EXHAUSTIVE` | ~10s-minutes | Optimal | Offline preprocessing, fixed sizes |

**Recommendation for RF signal processing**: Use `FFTW_MEASURE` or `FFTW_PATIENT`. The planning overhead is amortized over thousands of signal FFTs.

### Batch Processing (Rank-2 Tensors)

For processing multiple signals stored in a rank-2 tensor `[num_signals × signal_length]`, use `create_many_c2c()` instead of creating multiple 1D plans:

```cpp
// Process batch of signals with a single plan
constexpr int N = 1024;        // Signal length
constexpr int batch = 100;     // Number of signals

auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));
auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));

// Copy tensor data to complex array
for (size_t row = 0; row < batch; ++row) {
    for (size_t col = 0; col < N; ++col) {
        in[row * N + col][0] = tensor[row, col];  // real
        in[row * N + col][1] = 0.0;                // imag
    }
}

// Create batch plan
int n_dims[] = {N};
auto plan = fft::PlanFactory::create_many_c2c(
    1,              // rank: 1D transforms
    n_dims,         // transform size
    batch,          // number of transforms
    in, nullptr, 1, N,   // input: stride=1, dist=N (row-major)
    out, nullptr, 1, N,  // output: same layout
    FFTW_FORWARD,
    FFTW_MEASURE
);

plan.execute();  // Computes all FFTs in one call
```

**Why batch is faster**:
- Single function call overhead vs N calls
- Better cache utilization
- FFTW can apply cross-signal optimizations

### Wisdom (Persistent Plan Optimization)

FFTW "wisdom" stores the results of `FFTW_MEASURE` planning. Save it to avoid re-benchmarking on every startup:

```cpp
// At startup: load previous optimizations
fft::load_wisdom("fftw_wisdom.dat");

// Create plans with FFTW_MEASURE (uses cached wisdom if available)
auto plan = fft::PlanFactory::create_1d_c2c(N, in, out, FFTW_FORWARD, FFTW_MEASURE);

// At shutdown: save optimizations for next run
fft::save_wisdom("fftw_wisdom.dat");
```

**Tip**: Wisdom is architecture-specific. Don't share wisdom files across different CPUs.

### Memory Alignment

FFTW performs best with SIMD-aligned memory:

```cpp
// GOOD: Use fftw_malloc (automatically aligned)
auto* data = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
// ... use data ...
fftw_free(data);

// ACCEPTABLE: Ensure 32-byte alignment for AVX
alignas(32) double data[N * 2];  // Interleaved real/imag

// BAD: Unaligned heap allocation
double* data = new double[N * 2];  // May not be aligned
```

### Integration with TensorBase

The `TensorBase::data()` method returns a contiguous pointer suitable for FFTW:

```cpp
using SignalTensor = TensorBase<double, std::extents<size_t, NUM_SIGNALS, SIGNAL_LENGTH>>;
SignalTensor signals;

// Access raw data for FFTW
double* raw = signals.data();

// Note: For complex FFTs, you need to copy to fftw_complex arrays
// or use real-to-complex transforms
```

### Power Spectrum for Feature Extraction

For RF signal classification, the power spectrum `|FFT(x)|²` is often more useful than raw FFT coefficients:

```cpp
auto plan = fft::PlanFactory::create_1d_c2c(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
plan.execute();

// Compute power spectrum for ML features
double power[N];
fft::compute_power_spectrum(out, power, N);

// power[] now contains |real² + imag²| for each frequency bin
```

## API Reference

### FFTPlan

RAII wrapper for `fftw_plan`. Automatically destroys the plan on destruction.

- `execute()` - Execute the plan with the arrays used during creation
- `execute_dft(in, out)` - Execute with different (same-sized) arrays
- `valid()` - Check if plan was successfully created

### PlanFactory

Static factory methods for creating FFT plans:

- `create_1d_c2c(n, in, out, sign, flags)` - 1D complex-to-complex
- `create_1d_r2c(n, in, out, flags)` - 1D real-to-complex
- `create_many_c2c(...)` - Batch 1D complex-to-complex
- `create_many_r2c(...)` - Batch 1D real-to-complex

### Utilities

- `load_wisdom(filename)` - Load cached plan optimizations
- `save_wisdom(filename)` - Save plan optimizations
- `compute_power_spectrum(fft_out, power_out, size)` - |FFT|²
- `compute_magnitude(fft_out, mag_out, size)` - |FFT|

## Cross-Compilation (RISC-V)

FFTW is currently disabled for RISC-V cross-compilation. To enable:

1. Build FFTW for RISC-V target
2. Set `FFTW_ROOT` to the installation directory
3. Modify CMakeLists.txt to use `find_library` with the cross-compiled library

This is tracked as future work for the embedded deployment.
