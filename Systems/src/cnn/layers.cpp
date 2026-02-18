#include "layers.hpp"

namespace cnn {

bool fftw_backend_available() {
#ifdef USE_FFTW
    return true;
#else
    return false;
#endif
}

}
