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

## Forward inference CLI

Run full forward inference in forced FP32 or forced FP16 from the same checkpoint:

```bash
cd Classification

python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp32

python3 scripts/forward_inference.py \
  dataset/radioml_2016.10a.npz \
  --checkpoint ../Systems/src/cnn/radioml_cnn_pytorch.pth \
  --precision fp16
```

The script prints:
- model/input dtype
- elapsed time
- samples/sec
- accuracy if the `.npz` contains `y`

You can also save predictions or logits for direct FP32 vs FP16 comparisons:

```bash
python3 scripts/forward_inference.py dataset/radioml_2016.10a.npz --precision fp32 --save-preds outputs/preds_fp32.npy --save-logits outputs/logits_fp32.npy
python3 scripts/forward_inference.py dataset/radioml_2016.10a.npz --precision fp16 --save-preds outputs/preds_fp16.npy --save-logits outputs/logits_fp16.npy
```
