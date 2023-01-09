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

make -j BAREMETAL_ONLY=1 $@

