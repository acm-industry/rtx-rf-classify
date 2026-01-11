cmake_minimum_required(VERSION 3.15)
project(expr_demo)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG stable
)

FetchContent_MakeAvailable(mdspan)

add_executable(expr_demo
    src/main.cpp
)

target_link_libraries(expr_demo PRIVATE mdspan)

target_compile_options(expr_demo PRIVATE
    -Wall -Wextra -Wpedantic -O3 -march=native
)
