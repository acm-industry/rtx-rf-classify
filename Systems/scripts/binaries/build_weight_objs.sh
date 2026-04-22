#!/usr/bin/env bash
#
# DEPRECATED: kept for back-compat with anyone still depending on the
# pre-built .o files in Systems/src/binaries/.
#
# The CMake build now embeds the raw .bin files directly via Systems/src/
# weights.S using `.incbin`, which is portable across Linux/macOS/RISC-V
# without invoking objcopy or hard-coding an output ELF triple.  This script
# only works on GNU systems with x86-64 objcopy and is unsuitable for the
# RISC-V/Ara cross build, which is why it was replaced.  Prefer rebuilding
# via `cmake --build` instead.

set -e

ARCH="elf64-x86-64"
MACH="i386:x86-64"

echo "Converting .bin files to ELF object files..."
echo "WARNING: build_weight_objs.sh is deprecated; CMake now embeds the .bin"
echo "         files via src/weights.S (.incbin).  This script exists only"
echo "         for back-compat with the pre-incbin flow."

for f in *.bin; do
    base=$(basename "$f" .bin)

    echo "  -> $f"

    # Step 1: convert raw binary to relocatable ELF object
    objcopy \
        --input binary \
        --output ${ARCH} \
        --binary-architecture ${MACH} \
        "$f" "${base}.o"

    # Step 2: rename generated symbols to clean names
    # GNU objcopy creates:
    #   _binary_<filename>_bin_start
    #   _binary_<filename>_bin_end
    #   _binary_<filename>_bin_size

    objcopy \
        --redefine-sym _binary_${base}_bin_start=${base}_start \
        --redefine-sym _binary_${base}_bin_end=${base}_end \
        --redefine-sym _binary_${base}_bin_size=${base}_size \
        "${base}.o"

done

echo "Done. Generated .o files:"
ls -1 *.o