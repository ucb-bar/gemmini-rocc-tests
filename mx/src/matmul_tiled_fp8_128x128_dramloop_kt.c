// dramloop_nc with native K-tiling: 2 chunks x 2 K-tiles of 64, accumulating loops (C on the last), loop scales.
#define NCHUNKS 2
#define K_TILES 2
#define LOOP_SCALES 1
#include "matmul_tiled_fp8_128x128_dramloop_nc.c"
