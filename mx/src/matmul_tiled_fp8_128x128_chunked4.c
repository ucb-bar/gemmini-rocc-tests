// 4-chunk variant of matmul_tiled_fp8_128x128_chunked (32 rows per chunk; chunks alternate the 2 acc banks).
#define NCHUNKS 4
#include "matmul_tiled_fp8_128x128_chunked.c"
