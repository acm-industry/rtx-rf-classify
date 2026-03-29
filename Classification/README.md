# Classification (RadioML CNN)

## FP32 and FP16 checkpoints

Training (e.g. in `cnn.ipynb`) can save a float32 checkpoint:

```python
torch.save(model.state_dict(), 'radioml_cnn_pytorch.pth')
```

Convert a copy to float16 **without overwriting** the original:

```bash
cd Classification
python scripts/convert_checkpoint_fp32_to_fp16.py radioml_cnn_pytorch.pth -o radioml_cnn_pytorch_fp16.pth
```

Load either file interchangeably from Python (run with `Classification` on `PYTHONPATH`, or from this directory):

```python
from model_utils import load_radioml_cnn

loaded = load_radioml_cnn("radioml_cnn_pytorch_fp16.pth", num_classes=11)
# or FP32: load_radioml_cnn("radioml_cnn_pytorch.pth", num_classes=11)

out = loaded.model(loaded.match_inputs(batch))
```

Use `load_checkpoint_into_model(your_model, path, ...)` if you construct the module yourself.

- `precision="auto"` keeps dtypes from the file (FP32 or FP16).
- `precision="fp32"` / `"fp16"` forces casts after load.

See `model_utils.py` for details.
