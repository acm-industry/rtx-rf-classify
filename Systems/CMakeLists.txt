cmake_minimum_required(VERSION 3.25)
project(classify LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Path to the raw .bin weight files; passed to the assembler so weights.S /
# baremetal_input.S can resolve their `.incbin "..."` directives without
# embedding absolute paths in source.
set(WEIGHTS_BIN_DIR "${CMAKE_SOURCE_DIR}/scripts/binaries")

include(FetchContent)

option(USE_BLAS    "Use BLAS (OpenBLAS preferred)" ON)
option(USE_FFTW    "Use FFTW3" OFF)
option(USE_HWY     "Use Google Highway SIMD" OFF)
option(BUILD_TESTS "Build unit tests (streams_test, ...)" OFF)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)

if(USE_HWY)
  FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(highway)
endif()

if(USE_BLAS)
  find_package(BLAS REQUIRED)

  add_library(blas_interface INTERFACE)
  target_link_libraries(blas_interface INTERFACE ${BLAS_LIBRARIES})
  target_compile_definitions(blas_interface INTERFACE USE_BLAS)

  # Apple's Accelerate framework ships cblas.h inside the framework bundle
  # rather than on the default include path, so plain `#include <cblas.h>`
  # fails to find it.  Locate the framework's Headers/ directory and add it
  # to the interface include path.
  if(APPLE AND BLAS_LIBRARIES MATCHES "Accelerate.framework")
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE _macos_sdk
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(_accelerate_cblas
      "${_macos_sdk}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
    )
    if(EXISTS "${_accelerate_cblas}/cblas.h")
      target_include_directories(blas_interface INTERFACE "${_accelerate_cblas}")
    else()
      message(WARNING
        "USE_BLAS requested with Apple Accelerate but cblas.h not found at "
        "${_accelerate_cblas}.  Build will likely fail; install Xcode CLT or "
        "configure with -DUSE_BLAS=OFF to fall back to the scalar gemv/gemm."
      )
    endif()
  endif()
endif()

if(USE_FFTW)
  find_package(PkgConfig QUIET)

  if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
  endif()

  if(FFTW3_FOUND)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW3_LIBRARIES}"
    )
  else()
    message(FATAL_ERROR
      "FFTW3 not found. Install fftw3-dev or provide FFTW3 via toolchain."
    )
  endif()
endif()

# Weights live in scripts/binaries/*.bin and are embedded directly via
# src/weights.S using `.incbin`.  The assembler picks up the right object
# format automatically (Mach-O on macOS, x86-64 ELF on Linux, riscv64 ELF
# under the Ara Docker toolchain), which is why we replaced the previous
# objcopy-based scripts/binaries/build_weight_objs.sh flow.  The classify
# executable only needs the .bin files to exist; no separate generation step.
set(_required_bins
  weights_conv_1d_l1.bin     weights_conv_1d_bias_l1.bin   weights_bn_1d_l1.bin
  weights_conv_1d_l2.bin     weights_conv_1d_bias_l2.bin   weights_bn_1d_l2.bin
  weights_conv_1d_l3.bin     weights_conv_1d_bias_l3.bin   weights_bn_1d_l3.bin
  weights_conv_1d_l4.bin     weights_conv_1d_bias_l4.bin   weights_bn_1d_l4.bin
  linear_mat_1.bin           linear_add_1.bin
  linear_mat_2.bin           linear_add_2.bin
)
set(_missing_bins "")
foreach(_bin IN LISTS _required_bins)
  if(NOT EXISTS "${WEIGHTS_BIN_DIR}/${_bin}")
    list(APPEND _missing_bins "${_bin}")
  endif()
endforeach()

if(_missing_bins)
  if(BUILD_TESTS)
    message(STATUS
      "Missing weight .bin files in ${WEIGHTS_BIN_DIR}: ${_missing_bins}; "
      "skipping classify target (BUILD_TESTS=ON, tests-only configure)."
    )
    set(BUILD_CLASSIFY OFF)
  else()
    message(FATAL_ERROR
      "Missing weight .bin files in ${WEIGHTS_BIN_DIR}: ${_missing_bins}. "
      "Run scripts/binaries/convert_pth_to_bin.py first, or pass "
      "-DBUILD_TESTS=ON to configure tests-only without weights."
    )
  endif()
else()
  set(BUILD_CLASSIFY ON)
endif()

if(BUILD_CLASSIFY)
  add_executable(classify
    src/main.cpp
    src/vector/vector_ops.cpp
    src/weights.S
  )

  target_include_directories(classify PRIVATE
    src
    src/vector
  )

  target_link_libraries(classify PRIVATE mdspan)

  # Tell both clang's integrated assembler and GNU as where to find the .bin
  # files referenced by .incbin in src/weights.S.  -I covers cpp / IAS;
  # -Wa,-I covers GNU as when invoked indirectly.
  set_source_files_properties(src/weights.S PROPERTIES
    COMPILE_OPTIONS "-I${WEIGHTS_BIN_DIR};-Wa,-I${WEIGHTS_BIN_DIR}"
  )

  if(USE_HWY)
    target_link_libraries(classify PRIVATE hwy)
    target_compile_definitions(classify PRIVATE USE_HWY)
  endif()

  if(USE_BLAS)
    target_link_libraries(classify PRIVATE blas_interface)
  endif()

  if(USE_FFTW)
    target_link_libraries(classify PRIVATE FFTW3::fftw3)
    target_compile_definitions(classify PRIVATE USE_FFTW)
  endif()

  if(MSVC)
    target_compile_options(classify PRIVATE /W4)
  else()
    target_compile_options(classify PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    )

    if(CMAKE_CROSSCOMPILING)
      target_link_options(classify PRIVATE -static)
    endif()
  endif()
endif()

# ── Unit tests (opt-in via -DBUILD_TESTS=ON) ───────────────────────────────
if(BUILD_TESTS)
  find_package(Threads REQUIRED)

  add_executable(streams_test
    src/tests/test_streams.cpp
  )
  target_include_directories(streams_test PRIVATE
    src
    src/vector
  )
  target_link_libraries(streams_test PRIVATE mdspan Threads::Threads)
  if(NOT MSVC)
    target_compile_options(streams_test PRIVATE
      -O2 -Wall -Wextra -Wpedantic
    )
  endif()
endif()

# ── Bare-metal MVP (host dry-run + optional RISC-V cross-build) ────────────
#
# The canonical bare-metal binary is built by Ara's apps/Makefile inside the
# Docker image (see Systems/apps/classify_mvp/ and verify_rtl_simulation.sh).
# The targets below are convenience helpers:
#
#   classify_baremetal_host  - always built when the fixture exists; same TU
#                              as the bare-metal entry point but using the
#                              host's std::printf instead of HTIF.  Useful as
#                              a 30-second smoke test for inference.h +
#                              baremetal_stream.h without needing the cross
#                              toolchain or Docker.
#
#   BUILD_BAREMETAL_RISCV    - opt-in cross build for riscv64-unknown-elf.
#                              Requires `-DCMAKE_TOOLCHAIN_FILE=cmake/
#                              riscv-llvm.cmake` to point CMake at Ara's
#                              clang/clang++.  Use this only when you want
#                              to iterate on the cross-compiled ELF without
#                              waiting for the full Docker pipeline.
option(BUILD_BAREMETAL_RISCV "Cross-build classify_baremetal_riscv via cmake/riscv-llvm.cmake" OFF)

set(_baremetal_input_bin "${CMAKE_SOURCE_DIR}/src/binaries/baremetal_input.bin")
if(EXISTS "${_baremetal_input_bin}")
  add_executable(classify_baremetal_host
    src/main_baremetal.cpp
    src/weights.S
    src/baremetal_input.S
  )
  target_include_directories(classify_baremetal_host PRIVATE src src/vector)
  target_link_libraries(classify_baremetal_host PRIVATE mdspan)
  set_source_files_properties(src/weights.S PROPERTIES
    COMPILE_OPTIONS "-I${WEIGHTS_BIN_DIR};-Wa,-I${WEIGHTS_BIN_DIR}"
  )
  set_source_files_properties(src/baremetal_input.S PROPERTIES
    COMPILE_OPTIONS "-I${CMAKE_SOURCE_DIR}/src/binaries;-Wa,-I${CMAKE_SOURCE_DIR}/src/binaries"
  )
  if(NOT MSVC)
    # Intentionally no -DUSE_BLAS so blas_ops.h takes the scalar gemv/gemm
    # fallback, mirroring the actual bare-metal build.  We do NOT pass
    # -fno-exceptions here because the bundled mdspan headers contain
    # `throw std::out_of_range(...)` in their (uninstantiated) at() overloads
    # which clang parses eagerly under -fno-exceptions and rejects.  The
    # genuine cross-compile to riscv64-unknown-elf goes through Ara's
    # apps/Makefile (Step 8), which tolerates this either by patching mdspan
    # or by using the GCC frontend that skips the parse-time check.
    target_compile_options(classify_baremetal_host PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O2>
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    )
  endif()
elseif(NOT BUILD_TESTS)
  message(STATUS
    "Skipping classify_baremetal_host (no fixture at ${_baremetal_input_bin}; "
    "run scripts/binaries/generate_baremetal_input.py to enable it)."
  )
endif()

if(BUILD_BAREMETAL_RISCV)
  if(NOT CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR
      "BUILD_BAREMETAL_RISCV=ON requires a RISC-V toolchain file. "
      "Reconfigure with -DCMAKE_TOOLCHAIN_FILE=cmake/riscv-llvm.cmake"
    )
  endif()

  add_executable(classify_baremetal_riscv
    src/main_baremetal.cpp
    src/weights.S
    src/baremetal_input.S
  )
  target_include_directories(classify_baremetal_riscv PRIVATE src src/vector)
  target_link_libraries(classify_baremetal_riscv PRIVATE mdspan)
  set_source_files_properties(src/weights.S PROPERTIES
    COMPILE_OPTIONS "-I${WEIGHTS_BIN_DIR};-Wa,-I${WEIGHTS_BIN_DIR}"
  )
  set_source_files_properties(src/baremetal_input.S PROPERTIES
    COMPILE_OPTIONS "-I${CMAKE_SOURCE_DIR}/src/binaries;-Wa,-I${CMAKE_SOURCE_DIR}/src/binaries"
  )
  target_compile_options(classify_baremetal_riscv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
    $<$<COMPILE_LANGUAGE:CXX>:-mcmodel=medany>
  )
endif()
