# RTX RISC-V RF Classification

## About

A RF classifier designed to leverage RISC-V vector instruction extensions. The project is twofold in designing an optimized RISC-V program for use in strained environments and in designing the RF classification network.

## Contributing
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guidelines.

## FP32 vs FP16 Inference Comparison (Commands)

### 0) Start in repo root
```bash
cd <your-repo-root>
# e.g. cd /path/to/rtx-rf-classify
```

### 1) Optional: convert the PyTorch checkpoint to FP16
This is only needed if you want a half-precision `.pth` for Python-side inference.

```bash
cd Classification
python3 scripts/convert_checkpoint_fp32_to_fp16.py ../Systems/src/cnn/radioml_cnn_pytorch.pth -o radioml_cnn_pytorch_fp16.pth
```

### 2) Python forward inference comparison (FP32 vs FP16)
This path compares the PyTorch model directly and forces the full forward pass to run in either FP32 or FP16.

```bash
cd Classification

# FP32 weights + FP32 activations
python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp32 \
  --batch-size 256

# FP32 checkpoint cast to full FP16 inference
python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp16 \
  --batch-size 256
```

If you want direct prediction/logit diffs from the same script:

```bash
cd Classification

python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp32 \
  --save-preds outputs/preds_fp32.npy \
  --save-logits outputs/logits_fp32.npy

python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp16 \
  --save-preds outputs/preds_fp16.npy \
  --save-logits outputs/logits_fp16.npy
```

Then compare:

```bash
python3 - <<'PY'
import numpy as np
p32 = np.load('Classification/outputs/preds_fp32.npy')
p16 = np.load('Classification/outputs/preds_fp16.npy')
l32 = np.load('Classification/outputs/logits_fp32.npy')
l16 = np.load('Classification/outputs/logits_fp16.npy')
print('argmax_agreement', np.mean(p32 == p16))
print('l2', np.linalg.norm(l32 - l16))
print('max_abs', np.max(np.abs(l32 - l16)))
PY
```

### 3) Notebook-based TCP inference comparison
1. Open `Classification/dataset/radioml.ipynb` in Jupyter.
2. Run cells to preprocess and generate `output` + `y` for the test dataset.
3. Run the socket helper and loop cells starting at:
   - `import socket as sock`
   - `def recv_exact(sock, n)`
   - send batch len and send `batch[i:i+BUNDLE_COUNT].tobytes()`
   - validate `np.frombuffer(data, dtype=np.uint8)` against `y`

This notebook does exactly the same per-sample TCP protocol as the embedded C++ service in `Systems/src/main.cpp`:
- 4 bytes: big-endian batch length
- then `batch_len` samples of shape `(3, 128)` in contiguous `float32`
- then `batch_len` output bytes, one `uint8` class index per sample

4. For FP32: start server in `Systems` with `build/classify`.
5. Run the notebook cells and record accuracy.
6. For FP16: start server in `Systems` with `build-fp16/classify` (after building with `-DWEIGHTS_FP16=ON`).
7. Run the same notebook cells, compare accuracies and timing.

### 4) Generate System weight blobs and object files (FP32 and FP16)
```bash
mkdir -p Systems/generated/weights-fp32 Systems/generated/weights-fp16

cd Systems/generated/weights-fp32
python3 ../../scripts/binaries/convert_pth_to_bin.py ../../src/cnn/radioml_cnn_pytorch.pth
bash ../../scripts/binaries/build_weight_objs.sh

cd ../weights-fp16
python3 ../../scripts/binaries/convert_pth_to_bin.py ../../src/cnn/radioml_cnn_pytorch.pth --fp16
bash ../../scripts/binaries/build_weight_objs.sh
```

### 5) Build C++ binaries
```bash
cd Systems

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWEIGHTS_FP16=OFF \
  -DWEIGHT_OBJECTS_DIR="$PWD/generated/weights-fp32"
cmake --build build -j

cmake -S . -B build-fp16 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWEIGHTS_FP16=ON \
  -DWEIGHT_OBJECTS_DIR="$PWD/generated/weights-fp16"
cmake --build build-fp16 -j

# real FP16 forward-pass storage/compute path
cmake -S . -B build-fp16-compute -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWEIGHTS_FP16=ON \
  -DCOMPUTE_FP16=ON \
  -DWEIGHT_OBJECTS_DIR="$PWD/generated/weights-fp16"
cmake --build build-fp16-compute -j
```

### 6) Run and compare with TCP client
```bash
# run this in terminal 1
cd /Users/simon/Documents/GitHub/rtx-rf-classify/Systems
build/classify      # FP32
# or
build-fp16/classify # FP16

# run this in terminal 2
cat > /tmp/tcp_infer.py <<'PY'
import socket, struct, numpy as np
B = 16
x = np.random.randn(B,3,128).astype(np.float32)
with socket.create_connection(('127.0.0.1',8080), timeout=10) as s:
    s.sendall(struct.pack('>I',B))
    s.sendall(x.tobytes())
    preds = s.recv(B)
print('preds', list(preds))
PY
python3 /tmp/tcp_infer.py
```

### 7) Accuracy comparison
- Compare `accuracy=` from `Classification/scripts/forward_inference.py` for `--precision fp32` vs `--precision fp16` to measure full PyTorch FP32 vs FP16 inference.
- Compare end-to-end classification accuracy from the notebook for `build/classify` vs `build-fp16/classify`.
- The current TCP service returns only `uint8` argmax predictions, not the 11 logits, so L2 / max-abs logit comparisons are not available unless you instrument the server to send logits.

### 8) Timing
- Warm up, then run repeated batches in both modes and average. 
- FP16 has one-time decode overhead at startup; steadystate is the main comparison.
- `COMPUTE_FP16=ON` changes the TCP payload to FP16 samples (`3 * 128 * 2` bytes per sample) because the inference buffers run in half precision.
