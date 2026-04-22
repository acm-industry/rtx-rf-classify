# Per-app Makefile for the classification MVP under pulp-platform/ara.
#
# Ara's top-level apps/Makefile expects each subdirectory to either drop a
# main.c/main.cpp into the dir (which it picks up by wildcard), or supply a
# per-app Makefile fragment that overrides defaults.  We do the latter so we
# can:
#
#   - List the additional .S files (weights.S, baremetal_input.S) that the
#     staging script copies in alongside main.cpp.
#   - Set -fno-exceptions / -fno-rtti / -mcmodel=medany flags appropriate to
#     the bare-metal target (Ara's runtime is built with these too).
#   - Inject -DUSE_BLAS *off* (i.e. nothing) so blas_ops.h takes the scalar
#     fallback path (no libopenblas in the bare-metal newlib sysroot).
#   - Pass -DBAREMETAL_USE_ARA_PRINTF=1 so baremetal_stream.h emits HTIF
#     output via Ara's printf.h instead of host <cstdio>.
#
# Ara's per-app Makefile is included from apps/Makefile and runs *before* the
# default rules, so we only export variables here - we don't define recipes.
# If the upstream Ara doesn't honour these exact variable names, the staging
# script (verify_rtl_simulation.sh) is the right place to translate them.

app_name := classify_mvp

# Sources picked up by the wildcard in apps/Makefile.  We list them
# explicitly so the staging script can sanity-check that everything got
# copied in before invoking make.
app_srcs := \
    main.cpp \
    weights.S \
    baremetal_input.S

# Extra flags merged into the per-language CFLAGS / CXXFLAGS / ASFLAGS that
# Ara's runtime.mk derives from the toolchain config.  We avoid touching
# the linker flags here; Ara's runtime drives those.
EXTRA_CFLAGS   ?=
EXTRA_CXXFLAGS ?= -std=c++23 -fno-exceptions -fno-rtti -O3
EXTRA_ASFLAGS  ?= -I.

EXTRA_DEFINES  ?= -DBAREMETAL_USE_ARA_PRINTF=1

# Helpful when debugging staging: print what the runner copied in.
.PHONY: classify_mvp_describe
classify_mvp_describe:
	@echo "[classify_mvp] sources : $(app_srcs)"
	@echo "[classify_mvp] cxxflags: $(EXTRA_CXXFLAGS)"
	@echo "[classify_mvp] defines : $(EXTRA_DEFINES)"
