#!/usr/bin/env bash

if [ ! -d "build_spike" ] ; then
    autoconf && \
        mkdir build_spike && cd build_spike && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build_spike
./config.status > /dev/null   # pick up Makefile.in changes (e.g. the mx target)

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]] ; then
    make -j RUNNER=spike $@
else
    make -j RUNNER=spike BAREMETAL_ONLY=1 $@
fi
