/**
 * Ara apps/<name>/main.cpp shim for the bare-metal classification MVP.
 *
 * The host runner (Systems/scripts/verify_rtl_simulation.sh, --app mvp) stages
 * this directory into ${ARA_DIR}/apps/classify_mvp/ along with all the
 * Systems/src/ headers and .S files it depends on.  Ara's apps/Makefile then
 * compiles this main.cpp with its bare-metal runtime, links against the
 * staged weights.S + baremetal_input.S, and produces bin/classify_mvp.
 *
 * This file deliberately does almost nothing beyond:
 *   1. Switching baremetal_stream.h over to Ara's printf-via-HTIF path.
 *   2. Pulling in main_baremetal.cpp as a single translation unit so Ara's
 *      Makefile only has to know about main.cpp (it doesn't need to be
 *      taught about the broader Systems/src layout).
 *
 * Anything app-specific belongs in main_baremetal.cpp; this shim should stay
 * trivial enough that the staging script never needs to touch it.
 */

#define BAREMETAL_USE_ARA_PRINTF 1

#include "main_baremetal.cpp"
