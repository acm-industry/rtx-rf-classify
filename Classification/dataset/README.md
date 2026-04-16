# Dataset parsing notes

The dataset in this workspace is not image data yet. It is radio IQ data.

## What is in each folder

- `csoi/`
  - Contains `.mat` files such as `csoi_0_matfile.mat`.
  - These already store complex IQ directly in keys like `csoi_ComplexVoltageData`.
  - One file currently looks like `2,500,000` complex samples at `1.25e9` Hz.

- `Scenario A_1 (easy)/IQ`
  - Contains `128` `.tmp` files named like `scenario_A_1_<rx_index>_<channel_index>_0.tmp`.
  - These are MIDAS Blue files with complex `int16` IQ (`CI`) and a 512-byte header.
  - Based on the metadata, `32` receiver positions times `4` channels = `128` captures.

- `Scenario A_1 (easy)/IQ_No_Interferers`
  - Same receiver layout, but without interferers.
  - This is useful if you want a binary task like `clean` vs `with_interference`.

- `Scenario A_1 (easy)/scenario_A_1.json`
  - Contains the scene description.
  - It lists the interferers and the `RxVehicle_*` receiver positions.

## CNN-ready shape

Most RF CNN pipelines do not train on the raw complex array directly. A common first step is:

1. Read complex IQ.
2. Split into I and Q channels.
3. Window the signal into fixed-length chunks.
4. Train on tensors shaped like `(num_examples, 2, window_size)`.

That is exactly what `rf_dataset_loader.py` does.

## Quick start

Run:

```powershell
python .\rtx-rf-classify\Classification\dataset\rf_dataset_loader.py
```

In Python:

```python
from pathlib import Path
from Classification.dataset.rf_dataset_loader import (
    build_scenario_windows,
    list_scenario_a1_captures,
    load_csoi_mat,
)

root = Path(r"C:\Users\ktan0\Documents\rtx")

csoi = load_csoi_mat(root / "csoi" / "csoi_0_matfile.mat")
print(csoi["iq"].shape)

captures = list_scenario_a1_captures(
    root / "Scenario A_1 (easy)" / "IQ",
    root / "Scenario A_1 (easy)" / "scenario_A_1.json",
)

X, meta = build_scenario_windows(captures[:2], window_size=1024, stride=512)
print(X.shape)  # (N, 2, 1024)
print(meta[0])
```

## Label ideas

The correct label design depends on your goal:

- Signal classification:
  - Use `csoi`, `lte`, `nr`, `rtu`, etc. as classes.
  - For this you still need the matching source `.mat` files for each emitter class, not only `csoi`.

- Binary interference detection:
  - Label `Scenario A_1 (easy)/IQ` as `1`.
  - Label `Scenario A_1 (easy)/IQ_No_Interferers` as `0`.

- Receiver-position or channel prediction:
  - Use `rx_index` or `channel_index` from the filename as labels.
  - This is usually less useful unless that is the actual task.

## Important assumption

`rf_dataset_loader.py` assumes the filename index in `scenario_A_1_<rx>_<channel>_0.tmp` matches the order of `RxVehicle_*` entries in `scenario_A_1.json`. That assumption matches the counts and ordering in this dataset, but if you have external documentation that says otherwise, we should update the mapping.
