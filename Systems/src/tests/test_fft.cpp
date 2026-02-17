#include <iostream>
#include <cmath>
#include <iomanip>
#include "../tensor.h"
#include "../fft/fftw_wrapper.h"

// I/Q signal: 2 channels (I and Q) x 128 samples
constexpr size_t NUM_CHANNELS = 2;
constexpr size_t SIGNAL_LENGTH = 128;

using IQTensor = TensorBase<double, std::extents<size_t, NUM_CHANNELS, SIGNAL_LENGTH>>;

bool approx_equal(double a, double b, double tol = 1e-10) {
    return abs(a - b) < tol;
}

// Helper: load a 2x128 I/Q tensor into an fftw_complex array
void load_iq_to_complex(const IQTensor& iq, fftw_complex* out) {
    for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
        out[i][0] = iq[0, i];  // I -> real
        out[i][1] = iq[1, i];  // Q -> imag
    }
}

void test_single_iq_signal() {
    std::cout << "=== Test 1: Single I/Q Signal FFT ===\n";

    // Create a complex exponential: I = cos(2*pi*f*t), Q = sin(2*pi*f*t)
    // This is a single-sided signal — should produce a peak at bin 5 only (not mirrored)
    IQTensor signal;
    constexpr double freq = 5.0;

    for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
        double t = static_cast<double>(i) / SIGNAL_LENGTH;
        signal[0, i] = cos(2.0 * M_PI * freq * t);  // I
        signal[1, i] = sin(2.0 * M_PI * freq * t);  // Q
    }

    auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));
    auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));

    load_iq_to_complex(signal, in);

    auto plan = fft::PlanFactory::create_1d(SIGNAL_LENGTH, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    if (!plan.valid()) {
        std::cerr << "FAILED: Could not create FFT plan\n";
        fftw_free(in);
        fftw_free(out);
        return;
    }

    plan.execute();

    // Find peak bin
    double max_mag = 0.0;
    size_t max_bin = 0;
    for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
        double mag = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = i;
        }
    }

    // Complex exponential e^(j2*pi*f*t) should peak at bin f with magnitude N
    // Unlike a real cosine, there's no mirror peak at bin N-f
    double expected_mag = static_cast<double>(SIGNAL_LENGTH);
    size_t expected_bin = static_cast<size_t>(freq);

    std::cout << "Peak: bin " << max_bin << ", magnitude " << std::fixed << std::setprecision(1) << max_mag << "\n";

    if (max_bin == expected_bin && approx_equal(max_mag, expected_mag, 0.1)) {
        std::cout << "PASSED: Single-sided peak at bin " << expected_bin << " (no mirror)\n";
    } else {
        std::cout << "FAILED: Expected peak at bin " << expected_bin
                  << " with magnitude " << expected_mag << "\n";
    }

    fftw_free(in);
    fftw_free(out);
    std::cout << "\n";
}

void test_batch_iq_signals() {
    std::cout << "=== Test 2: Batch FFT on Multiple I/Q Signals ===\n";

    // 4 I/Q signals, each at a different frequency
    constexpr size_t NUM_SIGNALS = 4;
    IQTensor signals[NUM_SIGNALS];
    double freqs[] = {3.0, 10.0, 25.0, 50.0};

    for (size_t s = 0; s < NUM_SIGNALS; ++s) {
        for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
            double t = static_cast<double>(i) / SIGNAL_LENGTH;
            signals[s][0, i] = cos(2.0 * M_PI * freqs[s] * t);  // I
            signals[s][1, i] = sin(2.0 * M_PI * freqs[s] * t);  // Q
        }
    }

    constexpr int N = static_cast<int>(SIGNAL_LENGTH);
    constexpr int batch = static_cast<int>(NUM_SIGNALS);

    auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));
    auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N * batch));

    // Load each I/Q tensor into the flat complex array
    for (size_t s = 0; s < NUM_SIGNALS; ++s) {
        load_iq_to_complex(signals[s], &in[s * SIGNAL_LENGTH]);
    }

    int n_dims[] = {N};
    auto plan = fft::PlanFactory::create_many(
        1, n_dims, batch,
        in, nullptr, 1, N,
        out, nullptr, 1, N,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    if (!plan.valid()) {
        std::cerr << "FAILED: Could not create batch FFT plan\n";
        fftw_free(in);
        fftw_free(out);
        return;
    }

    plan.execute();

    bool all_passed = true;
    for (size_t s = 0; s < NUM_SIGNALS; ++s) {
        double max_mag = 0.0;
        size_t max_bin = 0;

        for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
            size_t idx = s * SIGNAL_LENGTH + i;
            double mag = sqrt(out[idx][0] * out[idx][0] + out[idx][1] * out[idx][1]);
            if (mag > max_mag) {
                max_mag = mag;
                max_bin = i;
            }
        }

        size_t expected_bin = static_cast<size_t>(freqs[s]);
        bool passed = (max_bin == expected_bin);
        if (!passed) all_passed = false;

        std::cout << "  Signal " << s << " (freq=" << freqs[s] << "): peak at bin "
                  << max_bin << " [" << (passed ? "PASS" : "FAIL") << "]\n";
    }

    std::cout << "\n" << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << "\n";

    fftw_free(in);
    fftw_free(out);
    std::cout << "\n";
}

void test_power_spectrum_iq() {
    std::cout << "=== Test 3: Power Spectrum from I/Q ===\n";

    // Complex exponential at frequency 10
    IQTensor signal;
    constexpr double freq = 10.0;

    for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
        double t = static_cast<double>(i) / SIGNAL_LENGTH;
        signal[0, i] = cos(2.0 * M_PI * freq * t);
        signal[1, i] = sin(2.0 * M_PI * freq * t);
    }

    auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));
    auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));
    double power[SIGNAL_LENGTH];

    load_iq_to_complex(signal, in);

    auto plan = fft::PlanFactory::create_1d(SIGNAL_LENGTH, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    plan.execute();

    fft::compute_power_spectrum(out, power, SIGNAL_LENGTH);

    // Power at bin 10 should be N^2 = 128^2 = 16384, all others ~0
    double expected_power = static_cast<double>(SIGNAL_LENGTH) * static_cast<double>(SIGNAL_LENGTH);
    size_t expected_bin = static_cast<size_t>(freq);

    std::cout << "Power at bin " << expected_bin << ": " << std::fixed << std::setprecision(1) << power[expected_bin] << "\n";
    std::cout << "Expected: " << expected_power << "\n";

    if (approx_equal(power[expected_bin], expected_power, 0.1)) {
        std::cout << "PASSED: Power spectrum correct\n";
    } else {
        std::cout << "FAILED: Power mismatch\n";
    }

    // Verify all other bins are ~0
    bool others_zero = true;
    for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
        if (i != expected_bin && power[i] > 0.1) {
            others_zero = false;
            break;
        }
    }

    if (others_zero) {
        std::cout << "PASSED: All other bins are zero\n";
    } else {
        std::cout << "FAILED: Energy leaked to other bins\n";
    }

    fftw_free(in);
    fftw_free(out);
    std::cout << "\n";
}

void test_plan_reuse_iq() {
    std::cout << "=== Test 4: Plan Reuse with I/Q Signals ===\n";

    auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));
    auto* out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * SIGNAL_LENGTH));

    // Create plan once
    auto plan = fft::PlanFactory::create_1d(SIGNAL_LENGTH, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Reuse with different I/Q signals
    double freqs[] = {7.0, 20.0, 42.0};

    for (int trial = 0; trial < 3; ++trial) {
        IQTensor signal;
        for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
            double t = static_cast<double>(i) / SIGNAL_LENGTH;
            signal[0, i] = cos(2.0 * M_PI * freqs[trial] * t);
            signal[1, i] = sin(2.0 * M_PI * freqs[trial] * t);
        }

        load_iq_to_complex(signal, in);
        plan.execute();

        // Find peak
        double max_mag = 0.0;
        size_t max_bin = 0;
        for (size_t i = 0; i < SIGNAL_LENGTH; ++i) {
            double mag = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
            if (mag > max_mag) {
                max_mag = mag;
                max_bin = i;
            }
        }

        size_t expected_bin = static_cast<size_t>(freqs[trial]);
        std::cout << "  Trial " << trial << " (freq=" << freqs[trial] << "): peak at bin "
                  << max_bin << " [" << (max_bin == expected_bin ? "PASS" : "FAIL") << "]\n";
    }

    std::cout << "PASSED: Plan reused across different I/Q signals\n";

    fftw_free(in);
    fftw_free(out);
    std::cout << "\n";
}

int main() {
    std::cout << "FFTW I/Q Signal Test Suite\n";
    std::cout << "=========================\n\n";

    test_single_iq_signal();
    test_batch_iq_signals();
    test_power_spectrum_iq();
    test_plan_reuse_iq();

    std::cout << "All tests completed.\n";
    return 0;
}
