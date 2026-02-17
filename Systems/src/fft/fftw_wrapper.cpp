#include "fftw_wrapper.h"
#include <cmath>

namespace fft {

FFTPlan PlanFactory::create_1d(int n, fftw_complex* in, fftw_complex* out,
                                    int sign, unsigned flags) {
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, sign, flags);
    return FFTPlan(plan);
}

FFTPlan PlanFactory::create_many(int rank, const int* n, int howmany,
                                      fftw_complex* in, const int* inembed,
                                      int istride, int idist,
                                      fftw_complex* out, const int* onembed,
                                      int ostride, int odist,
                                      int sign, unsigned flags) {
    fftw_plan plan = fftw_plan_many_dft(
        rank, n, howmany,
        in, inembed, istride, idist,
        out, onembed, ostride, odist,
        sign, flags
    );
    return FFTPlan(plan);
}

bool load_wisdom(const char* filename) {
    return fftw_import_wisdom_from_filename(filename) != 0;
}

bool save_wisdom(const char* filename) {
    return fftw_export_wisdom_to_filename(filename) != 0;
}

void compute_power_spectrum(const fftw_complex* fft_out, double* power_out,
                            std::size_t fft_size) {
    for (std::size_t i = 0; i < fft_size; ++i) {
        double real = fft_out[i][0];
        double imag = fft_out[i][1];
        power_out[i] = real * real + imag * imag;
    }
}

void compute_magnitude(const fftw_complex* fft_out, double* mag_out,
                       std::size_t fft_size) {
    for (std::size_t i = 0; i < fft_size; ++i) {
        double real = fft_out[i][0];
        double imag = fft_out[i][1];
        mag_out[i] = std::sqrt(real * real + imag * imag);
    }
}

}
