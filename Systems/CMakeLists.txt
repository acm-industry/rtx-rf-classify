cmake_minimum_required(VERSION 3.25)
project(classify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

option(USE_BLAS  "Use BLAS (OpenBLAS preferred)" ON)
option(USE_FFTW  "Use FFTW3" OFF)
option(USE_HWY   "Use Google Highway SIMD" OFF)

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
endif()

if(USE_FFTW)
  find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include /usr/include)
  find_library(FFTW3_LIBRARIES NAMES fftw3 PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  find_library(FFTW3F_LIBRARIES NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)

  if(FFTW3_INCLUDE_DIRS AND FFTW3_LIBRARIES AND FFTW3F_LIBRARIES)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    set_target_properties(FFTW3::fftw3 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${FFTW3_LIBRARIES};${FFTW3F_LIBRARIES}"
    )
  else()
    message(FATAL_ERROR
      "FFTW3 not found. Install fftw3-dev or provide FFTW3 via toolchain."
    )
  endif()
endif()

file(GLOB WEIGHT_OBJECTS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/binaries/*.o")

if(NOT WEIGHT_OBJECTS)
  message(FATAL_ERROR
    "No weight object files found in ${CMAKE_SOURCE_DIR}/src/binaries. "
    "Generate them first or fix WEIGHT_OBJECTS path."
  )
endif()

add_executable(classify
  src/main.cpp
  src/vector/vector_ops.cpp
  ${WEIGHT_OBJECTS}
)

target_include_directories(classify PRIVATE
  src
  src/vector
)

target_link_libraries(classify PRIVATE mdspan)

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
    -O3 -Wall -Wextra -Wpedantic -march=native -mtune=native
  )

  if(CMAKE_CROSSCOMPILING)
    target_link_options(classify PRIVATE -static)
  endif()
endif()

# Build the FFTW wrapper tests
add_executable(test_fftw_ops
  src/tests/test_fftw_ops.cpp
)

target_include_directories(test_fftw_ops PRIVATE
  src
  src/vector
)

target_link_libraries(test_fftw_ops PRIVATE mdspan)

if(USE_FFTW)
  target_link_libraries(test_fftw_ops PRIVATE FFTW3::fftw3)
  target_compile_definitions(test_fftw_ops PRIVATE USE_FFTW)
endif()

if(MSVC)
  target_compile_options(test_fftw_ops PRIVATE /W4)
else()
  target_compile_options(test_fftw_ops PRIVATE
    -O3 -Wall -Wextra -Wpedantic -march=native -mtune=native
  )
endif()
