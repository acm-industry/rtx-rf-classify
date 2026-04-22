# `classify_mvp` – bare-metal Ara application

This directory holds the **shim** files that pulp-platform/ara's
`apps/Makefile` consumes to build the classification MVP for the Ara core
(Verilator-simulated, bare-metal RISC-V).  Almost all of the actual code
lives in `Systems/src/`; the staging step performed by
`Systems/scripts/verify_rtl_simulation.sh --app mvp` copies the headers,
weight binaries, and input fixture into the Ara `apps/classify_mvp/`
directory before invoking `make`.

## Files

| File         | Purpose                                                              |
|--------------|----------------------------------------------------------------------|
| `main.cpp`   | Trivial shim; defines `BAREMETAL_USE_ARA_PRINTF` and includes        |
|              | `main_baremetal.cpp` so Ara only has to compile a single TU.         |
| `Makefile`   | Per-app fragment exporting `app_srcs`, `EXTRA_CXXFLAGS`, and the     |
|              | `BAREMETAL_USE_ARA_PRINTF` define.                                   |
| `README.md`  | This file.                                                           |

## Files staged in by the runner

After `verify_rtl_simulation.sh --app mvp` runs, you will also see (copied
out of `Systems/src/` and `Systems/scripts/binaries/`):

```
classify_mvp/
├── Makefile
├── README.md
├── main.cpp                         (this file)
├── main_baremetal.cpp               (from Systems/src/)
├── inference.h baremetal_stream.h streams.h tensor.h ...
├── weights.S baremetal_input.S
└── *.bin                            (the .incbin payloads)
```

## Wire format

* **Input** (baked into `baremetal_input.bin` at staging time):
  4-byte big-endian batch length, then `batch * 3 * 128` host-order floats.
* **Output** (printed to HTIF via Ara's `printf`):
  `[ARGMAX-START]<hh><hh>…<hh>[ARGMAX-END]\n[CYCLES] N\n`,
  one byte of argmax per sample.

The host runner extracts the bytes between the sentinels and diffs them
against `baremetal_expected.bin` (also staged in).
