#!/bin/bash

rm -rf ../../../data-collection-output
rm ../../../data-collection-vcs.sh
rm ../../../data-collection-spike.sh
cp og_baremetal_Makefile ../bareMetalC/Makefile
cd ..
./build.sh clean
cd gemmini-data-collection
rm ../bareMetalC/tiled_matmul_ws_perf-128_128_128.c
rm ../bareMetalC/tiled_matmul_ws_perf-512_32_512.c
rm ../bareMetalC/tiled_matmul_ws_perf-512_512_512.c
rm ../bareMetalC/tiled_matmul_ws_perf-1024_1024_1024.c
rm ../bareMetalC/conv-perf_224-3-64-7-2-3.c
rm ../bareMetalC/conv-perf_56-64-64-1-1-1.c
rm ../bareMetalC/conv-perf_14-256-256-3-1-1.c
rm ../bareMetalC/conv-perf_7-512-512-3-1-1.c
