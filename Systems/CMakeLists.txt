cmake_minimum_required(VERSION 3.26)
project(hello_riscv_qemu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supports both native (x86/x86_64) and cross-compilation (RISC-V)
# For native builds: cmake -S . -B build/native
# For RISC-V builds: cmake -S . -B build/riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake

add_executable(hello src/main.cpp)

# Use MSVC-specific warning flags on Windows, GCC/Clang flags otherwise
if (MSVC)
  target_compile_options(hello PRIVATE /W4)
else()
  target_compile_options(hello PRIVATE -Wall -Wextra -Wpedantic)
  # Static linking for cross-compilation (RISC-V); optional for native builds
  # Only apply static linking when cross-compiling (toolchain file is set)
  if(CMAKE_CROSSCOMPILING)
    target_link_options(hello PRIVATE -static)
  endif()
endif()
