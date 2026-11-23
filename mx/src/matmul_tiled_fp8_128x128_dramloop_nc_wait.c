// dramloop_nc with 2-D scale loads and NO fence: CONFIG_SCALE_MEM waits in HW (rs2[16]).
#define SCALES_2D 1
#define SCALE_WAIT 1
#include "matmul_tiled_fp8_128x128_dramloop_nc.c"
