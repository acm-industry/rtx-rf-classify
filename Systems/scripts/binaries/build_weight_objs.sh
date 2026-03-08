#!/usr/bin/env bash
set -e

ARCH="elf64-x86-64"
MACH="i386:x86-64"

echo "Converting .bin files to ELF object files..."

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