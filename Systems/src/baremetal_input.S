/*
 * baremetal_input.S - Embed the bake fixture for the bare-metal Ara MVP.
 *
 * Mirrors weights.S but for the single fixture file produced by
 * scripts/binaries/generate_baremetal_input.py.  The build system passes
 * -I .../src/binaries so the relative `.incbin "baremetal_input.bin"`
 * resolves at assembly time.
 *
 * Two symbols are exposed (with the usual __APPLE__ underscore mangling):
 *   baremetal_input_start[]   - first byte of the wire payload
 *   baremetal_input_end[]     - one-past-last byte; (end - start) == size
 *
 * main_baremetal.cpp wraps these in a StaticBufferIStream / BareIOStream
 * which then drives serve_inference exactly the same as the TCP build.
 */

#if defined(__APPLE__)
#  define G_START(name)   .global _##name##_start
#  define G_END(name)     .global _##name##_end
#  define LBL_START(name) _##name##_start:
#  define LBL_END(name)   _##name##_end:
#  define RODATA_SECTION  .section __DATA,__const
#else
#  define G_START(name)   .global name##_start
#  define G_END(name)     .global name##_end
#  define LBL_START(name) name##_start:
#  define LBL_END(name)   name##_end:
#  define RODATA_SECTION  .section .rodata
#endif

    RODATA_SECTION
    .balign 32

G_START(baremetal_input)
G_END(baremetal_input)
LBL_START(baremetal_input)
    .incbin "baremetal_input.bin"
LBL_END(baremetal_input)
