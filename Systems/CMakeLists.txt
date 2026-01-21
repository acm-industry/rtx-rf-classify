cmake_minimum_required(VERSION 3.26)
project(hello_riscv_qemu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hello src/main.cpp)

if (MSVC)
  target_compile_options(hello PRIVATE /W4)
else()
  target_compile_options(hello PRIVATE -Wall -Wextra -Wpedantic)
  # Static linking: linker flag, not compiler flag
  target_link_options(hello PRIVATE -static)
endif()
