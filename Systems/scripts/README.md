# Build and Test Scripts

## Purpose

This directory contains automation scripts for building and testing the project across different architectures and environments.

## RVV Vectorization Verification (Spike)

`spike_build_and_run.sh` runs the full RVV verification pipeline. It builds a Docker image with Spike (the RISC-V ISA simulator from the [Ara](https://github.com/pulp-platform/ara) ecosystem), then executes `verify_rvv_vectorization.sh` inside the container.

```bash
cd Systems
bash scripts/spike_build_and_run.sh
```

The first run takes several minutes to compile Spike, pk, and OpenBLAS from source. Subsequent runs use the cached Docker image.

### What the pipeline verifies

The pipeline answers one question: **does the compiler convert our scalar C++ loops into RISC-V Vector (RVV) instructions, and do those instructions produce correct results?**

It compiles `src/tests/test_rvv_kernels.cpp` -- a self-contained file containing 8 kernel functions that mirror the project's real operations -- in two ways and checks both.

### Phase 1: Assembly analysis (compile-time proof)

The source is compiled to a `.s` assembly text file using `clang++ -O3 -march=rv64gcv -S`. The `-S` flag stops before the assembler runs, producing human-readable RISC-V assembly. The script then counts RVV instruction mnemonics in the assembly, grouped into categories:

| Category | What it means | Example instructions |
|---|---|---|
| **setup** | Configure vector register length for the element type and grouping | `vsetvli`, `vsetivli` |
| **load_store** | Move data between memory and vector registers | `vle32.v` (load), `vse32.v` (store) |
| **arith** | Element-wise floating-point arithmetic across a vector register | `vfadd.vv`, `vfmul.vf` |
| **fma** | Fused multiply-accumulate (multiply + add in one instruction, no rounding in between) | `vfmacc.vv`, `vfmadd.vf` |
| **reduce** | Collapse a vector register to a single scalar (e.g. sum all elements) | `vfredosum.vs`, `vfredusum.vs` |
| **integer** | Integer vector operations (loop counters, address arithmetic) | `vadd.vi`, `vsll.vi` |
| **compare** | Element-wise comparisons that produce a mask register | `vmflt.vf`, `vmfle.vv` |
| **mask** | Combine or invert mask registers (for conditional vector operations) | `vmand.mm`, `vmnot.m` |

A count of 0 across all categories means the compiler did not vectorize. The script also breaks down the count per kernel function so you can see which operations benefited.

### Phase 2: Runtime correctness (functional proof)

The same source is compiled to a statically-linked RISC-V binary (no `-S` this time -- full compilation through assembler and linker). This binary is then executed on Spike:

```
spike --isa=rv64gcv pk ./test_binary
```

- **Spike** decodes and executes every RISC-V instruction, including all RVV vector instructions. It is a functional ISA simulator -- the reference implementation of the RISC-V spec. It executes each instruction faithfully, one at a time, with no approximation or binary translation.
- **pk** (proxy kernel) is a minimal runtime that handles system calls. When the test binary calls `printf()`, glibc issues an `ecall` instruction, pk intercepts it, and proxies the `write()` to the host terminal through Spike's Host-Target Interface (HTIF).

Each kernel is run with deterministic pseudo-random input (seeded LCG) and the output checksum is compared to a reference value computed inline in `main()`. The `got`/`expected` output shows the checksum from the kernel under test (`got`) vs. the independently-computed reference (`expected`). The tolerance is 0.5 to account for floating-point reassociation from `-ffast-math` -- vector reduction sums elements in a different order than a scalar loop, which changes the rounding.

### Phase 3: Instruction trace (execution proof)

Spike is run again with the `-l` flag, which logs every executed instruction to stderr. The script counts how many of those 900k+ instructions are vector instructions and reports a ratio. This is different from Phase 1: Phase 1 counts instructions *in the binary* (static), Phase 3 counts instructions *actually executed* (dynamic). The ratio is low (~0.1%) because most executed instructions are startup/teardown code from glibc and pk, not the kernels themselves.

### Test kernels

`src/tests/test_rvv_kernels.cpp` contains 8 `__attribute__((noinline))` functions. Each is a plain C++ loop that Clang's auto-vectorizer can transform into RVV instructions:

| Kernel | Project equivalent | Operation |
|---|---|---|
| `kernel_add` | `vec::add` | `out[i] = a[i] + b[i]` |
| `kernel_dot` | `vec::dot` | `sum += a[i] * b[i]` |
| `kernel_matvec` | `vec::matvec` | Matrix-vector multiply |
| `kernel_fma` | Expression system `a + b * c` | `out[i] = a[i] + b[i] * c[i]` |
| `kernel_conv1d` | `Conv1DInPlace` | Sliding dot product with padding |
| `kernel_relu` | `ReLU` functor | `out[i] = max(0, in[i])` |
| `kernel_maxpool` | `MaxPool1DInPlace` | Strided window maximum |
| `kernel_linear` | Linear layer | `out[i] = bias[i] + sum(w[i][j] * x[j])` |

The `noinline` attribute prevents the compiler from inlining kernels into `main()`, which would remove their function labels from the assembly and make per-kernel counting impossible.

### Why Clang instead of GCC

GCC 13 (shipped with Ubuntu 24.04) supports the `-march=rv64gcv` flag but its RISC-V V backend lacks vector type mappings for `float`. The auto-vectorizer cannot represent a vector of floats in RVV registers, so it never fires. Clang 18 has had mature RVV auto-vectorization since LLVM 16 and is the compiler the Ara project itself uses.

### Docker image contents

`docker/Dockerfile.riscv-rvv` builds from Ubuntu 24.04 and installs:

- **Clang 18** for compiling with RVV auto-vectorization
- **riscv64-linux-gnu-gcc/g++** for the cross-compilation sysroot, headers, and linker
- **Spike** built from [riscv-software-src/riscv-isa-sim](https://github.com/riscv-software-src/riscv-isa-sim) (upstream; Ara contributed their RVV support here)
- **pk** built from [riscv-software-src/riscv-pk](https://github.com/riscv-software-src/riscv-pk) for syscall proxying
- **OpenBLAS** cross-compiled for RISC-V (static, generic target)

### CMake toolchain

`cmake/toolchains/riscv64-rvv.cmake` configures CMake for RISC-V cross-compilation with `-march=rv64gcv` (the V vector extension enabled). This can be used independently of the Spike pipeline to cross-compile the full project with vector instructions enabled.

## Native Build Support

The `native_build_and_run.sh` script provides a convenient way to build and test the project for the native host architecture (x86/x86_64):

1. **Clean build**: Removes any existing native build directory to avoid CMake cache issues
2. **Native compilation**: Configures CMake for the host architecture (no toolchain file)
3. **Execution**: Runs the compiled binary directly on the host system

This is useful for:
- Quick local development and testing
- Validating code changes before cross-compiling for RISC-V
- Debugging issues that may be architecture-specific

## Ara RTL Simulation

`rtl_build_and_run.sh` runs the RTL simulation pipeline. It builds a Docker image containing the full [Ara](https://github.com/pulp-platform/ara) hardware simulation environment (LLVM toolchain, Verilator, Verilated hardware model), then executes `verify_rtl_simulation.sh` inside the container.

```bash
cd Systems
bash scripts/rtl_build_and_run.sh                  # default: 4 lanes, no trace
bash scripts/rtl_build_and_run.sh --trace           # generate FST waveform
bash scripts/rtl_build_and_run.sh --config 8_lanes  # 8-lane Ara configuration
```

The first run takes 2-4 hours to build the LLVM toolchain, Verilator, and hardware model from source. Subsequent runs use the cached Docker image and complete in minutes.

### What the pipeline verifies

The pipeline answers: **do our auto-vectorized kernels execute correctly on the actual Ara hardware model, and how many cycles do they take?**

It compiles `src/tests/test_rvv_kernels_baremetal.c` -- a bare-metal C port of the same 8 kernels used by the Spike pipeline -- as an Ara app, then runs it on the Verilated CVA6 + Ara system.

### Phase 1: App compilation

The bare-metal kernel source is copied into Ara's `apps/rtx_kernels/main.c` directory. Ara's build system compiles it with its LLVM toolchain targeting `riscv64-unknown-elf` (bare-metal, no OS). The key override is `LLVM_V_FLAGS=""` which removes Ara's default `-fno-vectorize` flag, allowing the auto-vectorizer to fire on our loops.

### Phase 2: RTL simulation

The compiled ELF binary is loaded into the Verilated Ara hardware model:

```
V_ara_tb_verilator -l ram,apps/bin/rtx_kernels,elf
```

The Verilated model is a cycle-accurate C++ executable generated by Verilator from Ara's SystemVerilog RTL. It faithfully models the CVA6 scalar core dispatching vector instructions to Ara's vector lanes, including all pipeline hazards, memory latency, and register dependencies.

Each kernel is timed with `start_timer()` / `stop_timer()` using the hardware cycle counter CSR, giving exact cycle counts. Output is sent to the host terminal via the Host-Target Interface (HTIF).

### FST waveform traces

Pass `--trace` to generate FST waveform files that record every signal transition in the hardware during simulation. These can be viewed with GTKWave for detailed analysis of how Ara's vector lanes process each kernel.

### Bare-metal test kernel

`src/tests/test_rvv_kernels_baremetal.c` is a C port of `test_rvv_kernels.cpp` adapted for Ara's bare-metal runtime:

- Uses Ara's custom `printf.h` instead of `<cstdio>` (bare-metal has no glibc)
- Uses `runtime.h` for cycle-accurate timing via `start_timer()` / `stop_timer()` / `get_timer()`
- Same 8 kernel functions with `__attribute__((noinline))`
- Same correctness checks (checksum comparison with 0.5 tolerance for `-ffast-math` reassociation)

### Ara configurations

Ara supports 2, 4, 8, or 16 vector lanes. More lanes means more parallel execution but larger hardware. The default is 4 lanes. Pass `--config 2_lanes` (or `4_lanes`, `8_lanes`, `16_lanes`) to test different configurations.

## Files

- `spike_build_and_run.sh`: Builds the Spike Docker image and runs the RVV vectorization verification pipeline
- `verify_rvv_vectorization.sh`: Three-phase pipeline (assembly analysis, Spike execution, instruction trace) that runs inside the Docker container
- `rtl_build_and_run.sh`: Builds the Ara RTL Docker image and runs the hardware simulation pipeline
- `verify_rtl_simulation.sh`: Two-phase pipeline (app compilation, RTL simulation) that runs inside the Ara Docker container
- `native_build_and_run.sh`: Builds and runs the project for the native host architecture
