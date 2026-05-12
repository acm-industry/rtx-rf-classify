g++ -std=c++23 -DUSE_FFTW -I .. \
  -I /opt/homebrew/opt/fftw/include \
  cnn_example.cpp layers.cpp ../memorybuffer.cpp ../fft/fftw_wrapper.cpp \
  -L /opt/homebrew/opt/fftw/lib -lfftw3 \
  -O2 -o /tmp/cnn_test_fft

/tmp/cnn_test_fft