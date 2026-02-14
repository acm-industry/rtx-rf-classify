cmake_minimum_required(VERSION 3.25)
project(hello_riscv_qemu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both native (x86/x86_64) and cross-compilation (RISC-V)
# For native builds: cmake -S . -B build/native
# For RISC-V builds: cmake -S . -B build/riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake

option(USE_BLAS "Use OpenBLAS for vector/matrix operations" ON)

include(FetchContent)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG stable
)

FetchContent_MakeAvailable(mdspan)

# Vector operations library (scalar / Highway / BLAS backend)
add_library(vector_ops STATIC src/vector/vector_ops.cpp)
target_include_directories(vector_ops PUBLIC src/vector)

if(USE_BLAS)
  if(CMAKE_CROSSCOMPILING)
    if(DEFINED OPENBLAS_ROOT)
      set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
      set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
      find_library(BLAS_LIBRARIES openblas PATHS "${OPENBLAS_ROOT}/lib" NO_DEFAULT_PATH)
      find_path(BLAS_INCLUDE_DIRS cblas.h PATHS "${OPENBLAS_ROOT}/include" NO_DEFAULT_PATH)
      if(NOT BLAS_LIBRARIES OR NOT BLAS_INCLUDE_DIRS)
        message(FATAL_ERROR "OpenBLAS not found at OPENBLAS_ROOT=${OPENBLAS_ROOT}")
      endif()
    else()
      message(FATAL_ERROR "USE_BLAS=ON with cross-compilation requires OPENBLAS_ROOT to be set")
    endif()
  else()
    find_package(BLAS REQUIRED)
    find_path(BLAS_INCLUDE_DIRS cblas.h
      HINTS /usr/include /usr/local/include /opt/homebrew/include
            /usr/include/openblas /opt/homebrew/opt/openblas/include)
    if(NOT BLAS_INCLUDE_DIRS)
      message(FATAL_ERROR "cblas.h not found. Install OpenBLAS development headers.")
    endif()
  endif()

  target_compile_definitions(vector_ops PUBLIC USE_BLAS)
  target_include_directories(vector_ops PUBLIC ${BLAS_INCLUDE_DIRS})
  target_link_libraries(vector_ops PUBLIC ${BLAS_LIBRARIES})
endif()

add_executable(hello src/main.cpp)

target_link_libraries(hello PRIVATE mdspan vector_ops)

# Use MSVC-specific warning flags on Windows, GCC/Clang flags otherwise
if (MSVC)
  target_compile_options(hello PRIVATE /W4)
else()
  target_compile_options(hello PRIVATE -O3 -Wall -Wextra -Wpedantic -fno-exceptions -fno-rtti)
  target_compile_options(vector_ops PRIVATE -O3 -Wall -Wextra -Wpedantic -fno-exceptions -fno-rtti)
  # Static linking for cross-compilation (RISC-V); optional for native builds
  # Only apply static linking when cross-compiling (toolchain file is set)
  if(CMAKE_CROSSCOMPILING)
    target_link_options(hello PRIVATE -static)
  endif()
endif()
