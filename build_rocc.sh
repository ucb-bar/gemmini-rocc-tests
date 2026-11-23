#!/usr/bin/env bash

# Build the gemmini-rocc-tests with Gemmini commands issued as RoCC instructions (e.g. rocket) into build_rocc/.
#
# Usage: ./build_rocc.sh [make targets...]   (targets are top-level dirs, like build.sh)
#   ./build_rocc.sh mx            # build all MX tests
#   ./build_rocc.sh bareMetalC    # build the original Gemmini tests
#   ./build_rocc.sh               # build everything
# To rebuild a single test after the first run:
#   ( cd build_rocc/mx && make -f .../mx/Makefile \
#       matmul_tiled_fp8_64x64_requant-baremetal )

if [ ! -d "build_rocc" ] ; then
    autoconf && \
        mkdir build_rocc && cd build_rocc && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build_rocc
./config.status > /dev/null   # pick up Makefile.in changes (e.g. the mx target)

# The RoCC/Spike flows only consume -baremetal binaries, so build those only. This also skips the
# -linux/-pk variants of tests that don't link in this env (e.g. llama_mlp/llama_attention: expf, tohost_exit).
make -j BAREMETAL_ONLY=1 $@
