// dramloop_nc with LOOP-MANAGED scales (2 chunks): no MX_LOAD_SCALES / CONFIG_SCALE_MEM / fence.
#define NCHUNKS 2
#define LOOP_SCALES 1
#include "matmul_tiled_fp8_128x128_dramloop_nc.c"
