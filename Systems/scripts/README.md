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

## Files

- `spike_build_and_run.sh`: Builds the Spike Docker image and runs the RVV vectorization verification pipeline
- `verify_rvv_vectorization.sh`: Three-phase pipeline (assembly analysis, Spike execution, instruction trace) that runs inside the Docker container
- `native_build_and_run.sh`: Builds and runs the project for the native host architecture
