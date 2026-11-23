// dramloop_ls with ReLU on the accumulated output (store-path activation); golden = relu(C_out_bf16).
#define NCHUNKS 2
#define LOOP_SCALES 1
#define RELU_OUT 1
#include "matmul_tiled_fp8_128x128_dramloop_nc.c"
