#!/usr/bin/env bash

if [ ! -d "build" ] ; then
    autoconf && \
        mkdir build && cd build && \
        ../configure &&
        cd ..

    if [ $? -ne 0 ] ; then
        echo $0 failed
        exit 1
    fi
fi

cd build
./config.status > /dev/null   # pick up Makefile.in changes (e.g. the mx target)

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]] ; then
    make -j $@
else
    make -j BAREMETAL_ONLY=1 $@
fi

