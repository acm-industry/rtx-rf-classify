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

### 1) Convert FP32 checkpoint to FP16
```bash
cd Classification
python3 scripts/convert_checkpoint_fp32_to_fp16.py ../Systems/src/cnn/radioml_cnn_pytorch.pth -o radioml_cnn_pytorch_fp16.pth
```

### 2) Notebook-based TCP inference comparison (preferred)
1. Open `Classification/dataset/radioml.ipynb` in Jupyter.
2. Run cells to preprocess and generate `output` + `y` for the test dataset.
3. Run the socket helper and loop cells starting at:
   - `import socket as sock`
   - `def recv_exact(sock, n)`
   - send batch len and send `batch[i:i+BUNDLE_COUNT].tobytes()`
   - validate `np.frombuffer(data, dtype=np.uint8)` against `y`

This notebook does exactly the same per-sample TCP protocol as the embedded C++ service in `Systems/src/main.cpp` (batch_len + 3*128*4 bytes per sample).

4. For FP32: start server in `Systems` with `build/classify`.
5. Run the notebook cells and record accuracy.
6. For FP16: start server in `Systems` with `build-fp16/classify` (after building with `-DWEIGHTS_FP16=ON`).
7. Run the same notebook cells, compare accuracies and timing.

### 3) Generate System weight blobs (FP32 and FP16)
```bash
cd Systems/scripts/binaries
python3 convert_pth_to_bin.py ../src/cnn/radioml_cnn_pytorch.pth
python3 convert_pth_to_bin.py ../src/cnn/radioml_cnn_pytorch.pth --fp16
```

### 4) Build C++ binaries
```bash
cd Systems
rm -rf build build-fp16

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWEIGHTS_FP16=OFF
cmake --build build -j

cmake -S . -B build-fp16 -G Ninja -DCMAKE_BUILD_TYPE=Release -DWEIGHTS_FP16=ON
cmake --build build-fp16 -j
```

### 5) Run and compare with TCP client
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

### 6) Accuracy comparison (argmax/L2)
- Compare `np.argmax(out_fp32,1)` vs `np.argmax(out_fp16,1)`.
- Compare `np.linalg.norm(out_fp32 - out_fp16)` and `np.max(np.abs(out_fp32 - out_fp16))`.

### 7) Timing
- Warm up, then run repeated batches in both modes and average. 
- FP16 has one-time decode overhead at startup; steadystate is the main comparison.