set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

set(RISCV_MARCH "rv64gc")
set(RISCV_MABI  "lp64d")

set(CMAKE_C_FLAGS_INIT   "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")

# Cross-compiled OpenBLAS (built in Docker image)
set(OPENBLAS_ROOT "/opt/openblas-riscv" CACHE PATH "Path to cross-compiled OpenBLAS")

