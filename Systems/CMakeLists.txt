cmake_minimum_required(VERSION 3.25)

set(CLASSIFY_DEFAULT_RISCV_TOOLCHAIN ON CACHE BOOL
  "Use the bundled riscv64-linux-gnu toolchain file by default"
)

if(CLASSIFY_DEFAULT_RISCV_TOOLCHAIN AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_LIST_DIR}/cmake/toolchains/riscv64-linux-gnu.cmake"
    CACHE FILEPATH "CMake toolchain file"
  )
endif()

project(classify LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF)

if(UNIX AND NOT APPLE)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

include(FetchContent)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable
)
FetchContent_MakeAvailable(mdspan)

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

if(MSVC)
  target_compile_options(classify PRIVATE /W4 /EHs-c- /GR-)
  target_compile_definitions(classify PRIVATE _HAS_EXCEPTIONS=0)
else()
  target_compile_options(classify PRIVATE
    -O3 -Wall -Wextra -Wpedantic -fno-exceptions -fno-rtti
  )

  target_link_options(classify PRIVATE -static -static-libgcc -static-libstdc++)
endif()
