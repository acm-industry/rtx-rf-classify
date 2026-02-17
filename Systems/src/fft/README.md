# FFTW Wrapper

C++ wrapper around FFTW3 for FFT-based signal preprocessing in the RF classification pipeline. Converts raw I/Q signals into power spectra for CNN input.

## Installation

FFTW3 is required for native builds:

```bash
# macOS
brew install fftw

# Ubuntu/Debian
sudo apt install libfftw3-dev
```

## Build and Run

```bash
cd Systems
g++ -std=c++23 -O3 \
  -I src -I /opt/homebrew/include -I build/native/_deps/mdspan-src/include \
  src/fft/fftw_wrapper.cpp src/tests/test_fft.cpp \
  -L /opt/homebrew/lib -lfftw3 \
  -o test_fft && ./test_fft
```

## Pipeline

```
I/Q Tensor [2 x 128]        (row 0 = I, row 1 = Q)
    | load_iq_to_complex()
fftw_complex[128]            (I -> real, Q -> imag)
    | plan.execute()
fftw_complex[128]            (complex frequency bins)
    | compute_power_spectrum()
double[128]                  (power at each frequency bin)
    |
CNN (Conv1D)                 -> class label
```

## Usage

### Single I/Q Signal

```cpp
#include "fft/fftw_wrapper.h"

constexpr int N = 128;
auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

// Load I/Q data
for (int i = 0; i < N; ++i) {
    in[i][0] = iq_tensor[0, i];  // I -> real
    in[i][1] = iq_tensor[1, i];  // Q -> imag
}

// Create plan once, execute many times
auto plan = fft::PlanFactory::create_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
plan.execute();

// Extract power spectrum for CNN
double power[N];
fft::compute_power_spectrum(out, power, N);

fftw_free(in);
fftw_free(out);
```

### Batch Processing

For processing multiple I/Q signals at once:

```cpp
constexpr int N = 128;
constexpr int batch = 100;

auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));
auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));

// Load each I/Q tensor into the flat array
for (int s = 0; s < batch; ++s) {
    for (int i = 0; i < N; ++i) {
        in[s * N + i][0] = signals[s][0, i];  // I
        in[s * N + i][1] = signals[s][1, i];  // Q
    }
}

int n_dims[] = {N};
auto plan = fft::PlanFactory::create_many(
    1, n_dims, batch,
    in, nullptr, 1, N,
    out, nullptr, 1, N,
    FFTW_FORWARD, FFTW_MEASURE
);

plan.execute();  // All FFTs in one call
```

## Planning Flags

| Flag | Planning Time | Execution Speed | Use Case |
|------|---------------|-----------------|----------|
| `FFTW_ESTIMATE` | ~0ms | Good | Development and testing |
| `FFTW_MEASURE` | ~100ms-1s | Better | Production (plan once, execute many) |
| `FFTW_PATIENT` | ~1-10s | Best | Batch processing, known sizes |

Use `FFTW_FORWARD` for forward FFT, `FFTW_BACKWARD` for inverse.

## Wisdom

Save/load plan optimizations to avoid re-benchmarking on startup:

```cpp
fft::load_wisdom("fftw_wisdom.dat");    // load at startup
auto plan = fft::PlanFactory::create_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
fft::save_wisdom("fftw_wisdom.dat");    // save at shutdown
```

Wisdom is architecture-specific — don't share across different CPUs.

## API

### FFTPlan

Wrapper for `fftw_plan`. Automatically destroyed on scope exit.

- `execute()` — run FFT with the arrays used during plan creation
- `execute_dft(in, out)` — run with different arrays (same size/alignment)
- `valid()` — check if plan was created successfully

### PlanFactory

- `create_1d(n, in, out, sign, flags)` — single 1D FFT
- `create_many(rank, n, howmany, ...)` — batch 1D FFT for multiple signals

### Utilities

- `compute_power_spectrum(fft_out, power_out, size)` — |FFT|^2
- `compute_magnitude(fft_out, mag_out, size)` — |FFT|
- `load_wisdom(filename)` / `save_wisdom(filename)` — persistent plan cache

## Note on RISC-V

Pretty sure FFTW is unoptimized on RISC-V due to it having SIMD optimizations primarily targeted at x86 / ARM. Could be wrong though but if it is the case, we prob have to write our own fft (if fft is even worth)

