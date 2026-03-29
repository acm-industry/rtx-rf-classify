# Embedded network weights (Systems / `main.cpp`)

## Storage format

| Build | On-disk / `.rodata` | At runtime (inference) |
|--------|---------------------|-------------------------|
| **FP32** (default) | IEEE binary32 (`float`) | Same: pointers cast to linked blobs |
| **FP16** (`WEIGHTS_FP16=ON`) | IEEE binary16 (`uint16_t` per scalar) | Decoded once in `init_fp16_weights()` to **float** buffers; math unchanged |

Activations stay **float32** in both modes. FP16 only reduces **linked weight size** (~2× smaller) and adds a one-time decode cost at startup plus ~641 KiB RAM for decoded weights (same as always having FP32 in RAM).

## Tensor layout and sizes (element counts = number of floats after decode)

| Symbol (C++) | Elements | Notes |
|--------------|----------|--------|
| `weights_conv_1d_l1` | 64 × 3 × 5 = **960** | Conv1D weights |
| `weights_conv_1d_bias_l1` | **64** | |
| `weights_bn_1d_l1` | 4 × 64 = **256** | γ, β, running_mean, running_var |
| `weights_conv_1d_l2` | 128 × 64 × 5 = **40,960** | |
| `weights_conv_1d_bias_l2` | **128** | |
| `weights_bn_1d_l2` | 4 × 128 = **512** | |
| `weights_conv_1d_l3` | 128 × 128 × 3 = **49,152** | |
| `weights_conv_1d_bias_l3` | **128** | |
| `weights_bn_1d_l3` | **512** | |
| `weights_conv_1d_l4` | **49,152** | |
| `weights_conv_1d_bias_l4` | **128** | |
| `weights_bn_1d_l4` | **512** | |
| `linear_mat_1` | 128 × 128 = **16,384** | First FC |
| `linear_add_1` | **128** | Bias |
| `linear_mat_2` | 11 × 128 = **1,408** | 11 classes |
| `linear_add_2` | **11** | Bias |

**Total scalars:** 160,395.

| Precision | Bytes in ROM (approx.) |
|-----------|-------------------------|
| FP32 | 160,395 × 4 ≈ **641,580** (~627 KiB) |
| FP16 | 160,395 × 2 ≈ **320,790** (~313 KiB) |

The PyTorch checkpoint must use the **Conv1d** layout expected by [`scripts/binaries/convert_pth_to_bin.py`](../scripts/binaries/convert_pth_to_bin.py) (`features.*`, `classifier.*`). This differs from the Conv2d `RadioMLCNN` in `Classification/cnn.ipynb`.

---

## Build pipeline

1. Export `.bin` files from the matching `.pth` (run inside `scripts/binaries/` or pass paths accordingly):

   ```bash
   python convert_pth_to_bin.py /path/to/model.pth
   ```

   For **FP16** blobs (half-sized files):

   ```bash
   python convert_pth_to_bin.py /path/to/model.pth --fp16
   ```

2. Build relocatable objects and copy into `src/binaries/` (CMake globs `*.o`):

   ```bash
   cd scripts/binaries
   ./build_weight_objs.sh
   cp *.o ../../src/binaries/
   ```

3. Configure C++:

   - **FP32 weights (default):** `cmake -S . -B build`
   - **FP16 weights:** same `.o` files must come from **`--fp16` bins**, then:

     ```bash
     cmake -S . -B build -DWEIGHTS_FP16=ON
     ```

4. Build `classify` (and optional `test_fp16_decode` for the half→float routine).

---

## Comparing accuracy and speed (inference)

- **Accuracy:** Feed the **same** TCP batches to two builds (FP32 vs FP16 weights). Compare **argmax class** per sample or L∞/L2 diff on the 11 logits. Differences come only from weight quantization; the forward pass is still float32 math.
- **Speed:** Measure end-to-end time per batch or per sample on the target (RISC-V). FP16 adds **decode once at startup**; steady-state inference should be similar to FP32 unless memory bandwidth dominates.

---

## Implementation references

- Decode: [`src/fp16_decode.h`](../src/fp16_decode.h), tests: [`src/tests/test_fp16_decode.cpp`](../src/tests/test_fp16_decode.cpp)
- Weight pointers / `init_fp16_weights`: [`src/maweights.cpp`](../src/maweights.cpp)
- Startup: [`src/main.cpp`](../src/main.cpp) calls `init_fp16_weights()` when `WEIGHTS_FP16=1`
