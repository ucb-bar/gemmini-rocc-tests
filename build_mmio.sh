#!/usr/bin/env bash

# Build the gemmini-rocc-tests with Gemmini commands issued through Radiance's MMIO registers
# (-DGEMMINI_MMIO, include/gemmini_mmio.h) instead of RoCC. Outputs land in build_mmio/.
#
# Usage: ./build_mmio.sh [make targets...]   (targets are top-level dirs, like build.sh)
#   ./build_mmio.sh mx            # build all MX tests
#   ./build_mmio.sh bareMetalC    # build the original Gemmini tests
#   ./build_mmio.sh               # build everything
# To rebuild a single test after the first run:
#   ( cd build_mmio/mx && make -f .../mx/Makefile EXTRA_CFLAGS=-DGEMMINI_MMIO \
#       matmul_tiled_fp8_64x64_requant-baremetal )

if [ ! -d "build_mmio" ] ; then
    autoconf && \
        mkdir build_mmio && cd build_mmio && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build_mmio
./config.status > /dev/null   # pick up Makefile.in changes (e.g. the mx target)

# The rocket/spike flow only consumes -baremetal binaries, so build those only. This also skips the
# -linux/-pk variants of tests that don't link in this env (e.g. llama_mlp/llama_attention: expf, tohost_exit).
make -j EXTRA_CFLAGS=-DGEMMINI_MMIO BAREMETAL_ONLY=1 $@
