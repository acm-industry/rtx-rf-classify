#ifndef __FFTW_WRAPPER_H__
#define __FFTW_WRAPPER_H__

#include <cstddef>
#include <fftw3.h>

namespace fft {

/**
 * FFTW Wrapper
 *
 * FFTW follows a plan -> execute model for efficiency:
 * 1. Create a plan once (potentially expensive)
 * 2. Execute the plan many times (very fast)
 *
 * Plans encode the FFT algorithm, input/output array shapes, and optimizations.
 * Creating a plan with FFTW_MEASURE benchmarks multiple algorithms to find the
 * fastest for specific hardware and array sizes.
 * 
 * There are other plan creation flags (FFTW_ESTIMATE, FFTW_PATIENT, FFTW_EXHAUSTIVE) that
 * increase planning time for potentially faster execution. 
 */
class FFTPlan {
public:
    // default constructor
    FFTPlan() noexcept : plan(nullptr) {} // no exception used to allow move semantics without throwing
    FFTPlan(fftw_plan plan) noexcept : plan(plan) {}

    // destructor
    ~FFTPlan() { 
        if (plan) fftw_destroy_plan(plan);
    }

    // move constructor
    FFTPlan(FFTPlan&& other) noexcept : plan(other.plan) {
        other.plan = nullptr;
    }

    // move assignment
    FFTPlan& operator=(FFTPlan&& other) noexcept {
        if (this != &other) {
            if (plan) {
                fftw_destroy_plan(plan);
            }
            plan = other.plan;
            other.plan = nullptr;
        }
        return *this;
    }


    // Executes plan with the original arrays specified at plan creation.
    void execute() const {
        if (plan) fftw_execute(plan);
    }

    // Executes same plan with new input/output arrays but size and alignment must match the original plan.
    void execute_dft(fftw_complex* in, fftw_complex* out) const {
        if (plan) fftw_execute_dft(plan, in, out);
    }

    bool valid() const noexcept { return plan != nullptr; }
    fftw_plan get() const noexcept { return plan; }

private:
    fftw_plan plan;
};

/**
 * PlanFactory: Creates FFTW plans with configurable planning effort.
 *
 * EFFICIENCY NOTES:
 * ================
 *
 * Planning Flags (tradeoff: planning time vs execution speed):
 * - FFTW_ESTIMATE: Use heuristics, very fast planning (~0ms), decent execution
 * - FFTW_MEASURE:  Benchmark algorithms, slower planning (~100ms-1s), faster execution
 * - FFTW_PATIENT:  More thorough benchmarking, even slower planning, fastest execution
 * - FFTW_EXHAUSTIVE: Try everything, very slow planning, absolute fastest execution
 *
 * For RF signal processing with repeated FFTs on same-sized data:
 * - Use FFTW_MEASURE or FFTW_PATIENT for production (plan once, execute many times)
 * - Use FFTW_ESTIMATE during development/testing for faster iteration
 *
 * WISDOM (persistent plan optimization):
 * - fftw_export_wisdom_to_file() / fftw_import_wisdom_from_file()
 * - Save wisdom after FFTW_MEASURE planning, reload on subsequent runs
 * - Eliminates re-benchmarking overhead on startup
 *
 * MEMORY ALIGNMENT:
 * - Use fftw_malloc()/fftw_free() for SIMD-aligned memory
 * - Or ensure 16-byte alignment for SSE, 32-byte for AVX
 *
 * BATCH PROCESSING (for rank-2 tensors representing multiple signals):
 * - Use fftw_planmany_dft() instead of multiple 1D plans
 * - Processes N signals in one call, better cache utilization
 */
class PlanFactory {
public:
    /**
     * Create a 1D complex-to-complex FFT plan.
     *
     * @param n         Size of the transform
     * @param in        Input array (fftw_complex*)
     * @param out       Output array (can be same as in for in-place)
     * @param sign      FFTW_FORWARD (-1) or FFTW_BACKWARD (+1)
     * @param flags     Planning flags (FFTW_ESTIMATE, FFTW_MEASURE, etc.)
     */
    static FFTPlan create_1d(int n, fftw_complex* in, fftw_complex* out,
                                  int sign = FFTW_FORWARD,
                                  unsigned flags = FFTW_ESTIMATE);


    /**
     * Create a batch 1D FFT plan for processing multiple signals.
     * Ideal for rank-2 tensors where each row is a separate signal.
     *
     * @param rank      Dimensionality of transform (1 for 1D FFT)
     * @param n         Array of transform sizes (length = rank)
     * @param howmany   Number of transforms (batch size, e.g., number of rows)
     * @param in        Input array
     * @param inembed   Storage dimensions of input (NULL = same as n)
     * @param istride   Stride between consecutive elements within a transform
     * @param idist     Distance between first elements of consecutive transforms
     * @param out       Output array
     * @param onembed   Storage dimensions of output (NULL = same as n)
     * @param ostride   Stride between consecutive output elements
     * @param odist     Distance between first elements of consecutive outputs
     * @param sign      FFTW_FORWARD or FFTW_BACKWARD
     * @param flags     Planning flags
     *
     * Example for row-major rank-2 tensor [num_signals x signal_length]:
     *   n = {signal_length}
     *   howmany = num_signals
     *   istride = 1, idist = signal_length (contiguous rows)
     *   ostride = 1, odist = signal_length
     */
    static FFTPlan create_many(int rank, const int* n, int howmany,
                                    fftw_complex* in, const int* inembed,
                                    int istride, int idist,
                                    fftw_complex* out, const int* onembed,
                                    int ostride, int odist,
                                    int sign = FFTW_FORWARD,
                                    unsigned flags = FFTW_ESTIMATE);

};

/**
 * Load FFTW wisdom from a file. Returns true on success.
 * Wisdom contains cached plan optimizations from previous FFTW_MEASURE runs.
 */
bool load_wisdom(const char* filename);

/**
 * Save FFTW wisdom to a file. Returns true on success.
 * Call after creating plans with FFTW_MEASURE to cache optimizations.
 */
bool save_wisdom(const char* filename);

/**
 * Compute power spectrum: |FFT(x)|^2
 * Useful for spectral analysis of RF signals.
 *
 * @param fft_out    Complex FFT output
 * @param power_out  Power spectrum output (size = fft_size)
 * @param fft_size   Number of complex FFT outputs
 */
void compute_power_spectrum(const fftw_complex* fft_out, double* power_out,
                            size_t fft_size);

/**
 * Compute magnitude spectrum: |FFT(x)|
 *
 * @param fft_out    Complex FFT output
 * @param mag_out    Magnitude spectrum output (size = fft_size)
 * @param fft_size   Number of complex FFT outputs
 */
void compute_magnitude(const fftw_complex* fft_out, double* mag_out,
                       size_t fft_size);

} // namespace fft

#endif // __FFTW_WRAPPER_H__
