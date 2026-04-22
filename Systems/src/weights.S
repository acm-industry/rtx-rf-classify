/*
 * weights.S - Portable weight embedding via .incbin.
 *
 * Replaces the old `objcopy --output elf64-x86-64` step in
 * scripts/binaries/build_weight_objs.sh.  The native assembler embeds the raw
 * .bin files for whatever target it is compiling for (x86-64 ELF on Linux,
 * Mach-O on macOS, riscv64 ELF inside the Ara Docker image).  Same .S file,
 * three platforms.
 *
 * Symbol naming matches the existing `extern const unsigned char
 * <name>_start[]` declarations in maweights.cpp.  On Mach-O, C symbols carry
 * a leading underscore in the object file; on ELF they do not.  We branch on
 * __APPLE__ to pick the right linkage name.
 *
 * The .bin files are looked up via the assembler's include search path, set
 * by the build system to Systems/scripts/binaries/ (passed as -I... and
 * -Wa,-I... so both clang's integrated assembler and GNU as find them).
 *
 * NB: every directive must be on its own physical line.  Earlier revisions
 * tried to wrap the whole entry in a function-like macro with `\` line
 * continuations, but the resulting `.global x; label: .incbin "f"; end:`
 * one-liner silently fails on clang's Mach-O backend (reports success but
 * emits zero-sized __const).  Multiple physical lines avoid the issue and
 * cost nothing in maintenance.
 */

#if defined(__APPLE__)
/* Mach-O: C externs carry a leading underscore; .rodata is __DATA,__const. */
#  define G_START(name)   .global _##name##_start
#  define G_END(name)     .global _##name##_end
#  define LBL_START(name) _##name##_start:
#  define LBL_END(name)   _##name##_end:
#  define RODATA_SECTION  .section __DATA,__const
#else
/* ELF (Linux/RISC-V): bare symbols, .rodata. */
#  define G_START(name)   .global name##_start
#  define G_END(name)     .global name##_end
#  define LBL_START(name) name##_start:
#  define LBL_END(name)   name##_end:
#  define RODATA_SECTION  .section .rodata
#endif

    RODATA_SECTION
    .balign 32

G_START(weights_conv_1d_l1)
G_END(weights_conv_1d_l1)
LBL_START(weights_conv_1d_l1)
    .incbin "weights_conv_1d_l1.bin"
LBL_END(weights_conv_1d_l1)

G_START(weights_conv_1d_bias_l1)
G_END(weights_conv_1d_bias_l1)
LBL_START(weights_conv_1d_bias_l1)
    .incbin "weights_conv_1d_bias_l1.bin"
LBL_END(weights_conv_1d_bias_l1)

G_START(weights_bn_1d_l1)
G_END(weights_bn_1d_l1)
LBL_START(weights_bn_1d_l1)
    .incbin "weights_bn_1d_l1.bin"
LBL_END(weights_bn_1d_l1)

G_START(weights_conv_1d_l2)
G_END(weights_conv_1d_l2)
LBL_START(weights_conv_1d_l2)
    .incbin "weights_conv_1d_l2.bin"
LBL_END(weights_conv_1d_l2)

G_START(weights_conv_1d_bias_l2)
G_END(weights_conv_1d_bias_l2)
LBL_START(weights_conv_1d_bias_l2)
    .incbin "weights_conv_1d_bias_l2.bin"
LBL_END(weights_conv_1d_bias_l2)

G_START(weights_bn_1d_l2)
G_END(weights_bn_1d_l2)
LBL_START(weights_bn_1d_l2)
    .incbin "weights_bn_1d_l2.bin"
LBL_END(weights_bn_1d_l2)

G_START(weights_conv_1d_l3)
G_END(weights_conv_1d_l3)
LBL_START(weights_conv_1d_l3)
    .incbin "weights_conv_1d_l3.bin"
LBL_END(weights_conv_1d_l3)

G_START(weights_conv_1d_bias_l3)
G_END(weights_conv_1d_bias_l3)
LBL_START(weights_conv_1d_bias_l3)
    .incbin "weights_conv_1d_bias_l3.bin"
LBL_END(weights_conv_1d_bias_l3)

G_START(weights_bn_1d_l3)
G_END(weights_bn_1d_l3)
LBL_START(weights_bn_1d_l3)
    .incbin "weights_bn_1d_l3.bin"
LBL_END(weights_bn_1d_l3)

G_START(weights_conv_1d_l4)
G_END(weights_conv_1d_l4)
LBL_START(weights_conv_1d_l4)
    .incbin "weights_conv_1d_l4.bin"
LBL_END(weights_conv_1d_l4)

G_START(weights_conv_1d_bias_l4)
G_END(weights_conv_1d_bias_l4)
LBL_START(weights_conv_1d_bias_l4)
    .incbin "weights_conv_1d_bias_l4.bin"
LBL_END(weights_conv_1d_bias_l4)

G_START(weights_bn_1d_l4)
G_END(weights_bn_1d_l4)
LBL_START(weights_bn_1d_l4)
    .incbin "weights_bn_1d_l4.bin"
LBL_END(weights_bn_1d_l4)

G_START(linear_mat_1)
G_END(linear_mat_1)
LBL_START(linear_mat_1)
    .incbin "linear_mat_1.bin"
LBL_END(linear_mat_1)

G_START(linear_add_1)
G_END(linear_add_1)
LBL_START(linear_add_1)
    .incbin "linear_add_1.bin"
LBL_END(linear_add_1)

G_START(linear_mat_2)
G_END(linear_mat_2)
LBL_START(linear_mat_2)
    .incbin "linear_mat_2.bin"
LBL_END(linear_mat_2)

G_START(linear_add_2)
G_END(linear_add_2)
LBL_START(linear_add_2)
    .incbin "linear_add_2.bin"
LBL_END(linear_add_2)
