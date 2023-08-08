#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// #define NUM_LAYERS %NUM_LAYERS%
// #define BATCH_SIZE "%BATCH_SIZE%"
// #define IN_DIM "%IN_DIM%"
// #define IN_CHANNELS "%IN_CHANNELS%"
// #define OUT_CHANNELS "%OUT_CHANNELS%"
// #define KERNEL_DIM "%KERNEL_DIM%"
// #define PADDING 0
// #define KERNEL_DILATION "%KERNEL_DILATION%"
// #define STRIDE "%STRIDE%"

// #define BATCHES "%TILE_BATCHES%"
// #define OCOLS "%TILE_OCOLS%"
// #define OROWS "%TILE_OROWS%"
// #define OCHS "%TILE_OCHS%"
// #define KCOLS "%TILE_KCOLS%"
// #define KROWS "%TILE_KROWS%"
// #define KCHS "%TILE_KCHS%"
// #define PERM_STR "%PERM_STR%"

#define NO_BIAS 1

// #define OUT_DIM ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
// #define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
// #define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    static elem_t input[1][40][40][2048];
    // elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
    static acc_t bias[2048];
    // static elem_t output[1][128][128][64];
    static elem_t weights_mat[2048][2048];
    static elem_t output_mat[11000][2048];

    const int NUM_LAYERS = %NUM_LAYERS%;
    const int BATCH_SIZE_LIST[%NUM_LAYERS%] = {%BATCH_SIZE%};
    const int IN_DIM_LIST[%NUM_LAYERS%] = {%IN_DIM%};
    const int IN_CHANNELS_LIST[%NUM_LAYERS%] = {%IN_CHANNELS%};
    const int OUT_CHANNELS_LIST[%NUM_LAYERS%] = {%OUT_CHANNELS%};
    const int KERNEL_DIM_LIST[%NUM_LAYERS%] = {%KERNEL_DIM%};
    // const int PADDING = 0
    const int KERNEL_DILATION_LIST[%NUM_LAYERS%] = {%KERNEL_DILATION%};
    const int STRIDE_LIST[%NUM_LAYERS%] = {%STRIDE%};

    const int BATCHES_LIST[%NUM_LAYERS%] = {%TILE_BATCHES%};
    const int OCOLS_LIST[%NUM_LAYERS%] = {%TILE_OCOLS%};
    const int OROWS_LIST[%NUM_LAYERS%] = {%TILE_OROWS%};
    const int OCHS_LIST[%NUM_LAYERS%] = {%TILE_OCHS%};
    const int KCOLS_LIST[%NUM_LAYERS%] = {%TILE_KCOLS%};
    const int KROWS_LIST[%NUM_LAYERS%] = {%TILE_KROWS%};
    const int KCHS_LIST[%NUM_LAYERS%] = {%TILE_KCHS%};
    const int SPATIAL_OCHS_LIST[%NUM_LAYERS%] = {%SPATIAL_TILE_OCHS%};
    const int SPATIAL_KCHS_LIST[%NUM_LAYERS%] = {%SPATIAL_TILE_KCHS%};
    const char *PERM_STR_LIST[%NUM_LAYERS%] = {%PERM_STR%};

    for (int l = 0; l < NUM_LAYERS; l++) {
        const int BATCH_SIZE = BATCH_SIZE_LIST[l];
        const int IN_DIM = IN_DIM_LIST[l];
        const int IN_CHANNELS = IN_CHANNELS_LIST[l];
        const int OUT_CHANNELS = OUT_CHANNELS_LIST[l];
        const int KERNEL_DIM = KERNEL_DIM_LIST[l];
        const int PADDING = KERNEL_DIM / 2;
        const int KERNEL_DILATION = KERNEL_DILATION_LIST[l];
        const int STRIDE = STRIDE_LIST[l];

        const int BATCHES = BATCHES_LIST[l];
        const int OCOLS = OCOLS_LIST[l];
        const int OROWS = OROWS_LIST[l];
        const int OCHS = OCHS_LIST[l];
        const int KCOLS = KCOLS_LIST[l];
        const int KROWS = KROWS_LIST[l];
        const int KCHS = KCHS_LIST[l];
        const int SPATIAL_OCHS = SPATIAL_OCHS_LIST[l];
        const int SPATIAL_KCHS = SPATIAL_KCHS_LIST[l];
        const char* PERM_STR = PERM_STR_LIST[l];

        const int OUT_DIM = ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1);
        const int PATCH_SIZE = (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS);
        const int N_PATCHES = (BATCH_SIZE * OUT_DIM * OUT_DIM);

        // elem_t input[1][IN_DIM][IN_DIM][IN_CHANNELS] = malloc(sizeof(elem_t) * IN_DIM * IN_DIM * IN_CHANNELS);
        // // elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS];
        // acc_t bias[OUT_CHANNELS] = malloc(sizeof(acc_t) * OUT_CHANNELS);
        // elem_t output[1][OUT_DIM][OUT_DIM][OUT_CHANNELS] = malloc(sizeof(elem_t) * OUT_DIM * OUT_DIM * OUT_CHANNELS);
        // elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] = malloc(sizeof(elem_t) * PATCH_SIZE * OUT_CHANNELS);
        // elem_t output_mat[N_PATCHES][OUT_CHANNELS] = malloc(sizeof(elem_t) * N_PATCHES * OUT_CHANNELS);

        // elem_t *input = malloc(sizeof(elem_t) * IN_DIM * IN_DIM * IN_CHANNELS);
        // acc_t *bias = malloc(sizeof(acc_t) * OUT_CHANNELS);
        // elem_t *output = malloc(sizeof(elem_t) * OUT_DIM * OUT_DIM * OUT_CHANNELS);
        // elem_t *weights_mat = malloc(sizeof(elem_t) * PATCH_SIZE * OUT_CHANNELS);
        // elem_t *output_mat = malloc(sizeof(elem_t) * N_PATCHES * OUT_CHANNELS);

        // printf("Input dimension: %u\n\n", IN_DIM);

        // printf("Kernel dimensions: %u\n", KERNEL_DIM); 
        // printf("Output dimension: %u\n", OUT_DIM);
        // printf("Input channels: %u\n", IN_CHANNELS); 
        // printf("Output channels: %u\n", OUT_CHANNELS); 
        // printf("Batch size: %u\n", BATCH_SIZE);
        // printf("Stride: %u\n", STRIDE);
        // printf("Kernel dilation: %u\n", KERNEL_DILATION); 
        // printf("Padding: %u\n", PADDING);

        // printf("TILE R: %u\n", KCOLS);
        // printf("TILE S: %u\n", KROWS);
        // printf("Tile P: %u\n", OCOLS);
        // printf("Tile Q: %u\n", OROWS);
        // printf("Tile C: %u\n", KCHS);
        // printf("Tile K: %u\n", OCHS);
        // printf("Tile N: %u\n", BATCHES);
        // printf("Spatial tile C: %u\n", SPATIAL_KCHS);
        // printf("Spatial tile K: %u\n", SPATIAL_OCHS);
        // printf("Perm: %s\n", PERM_STR);

        printf("%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%u_%s\n", KERNEL_DIM, OUT_DIM, IN_CHANNELS, OUT_CHANNELS, 
               BATCH_SIZE, STRIDE, KERNEL_DILATION, PADDING, KCOLS, KROWS, OCOLS, OROWS, KCHS, OCHS, BATCHES, SPATIAL_KCHS, SPATIAL_OCHS, PERM_STR);

        gemmini_flush(0);
        // printf("Gemmini conv...\n");
        uint64_t start_gemmini = read_cycles();

        // assert((in_dim + 2*padding - kernel_dim) % stride == 0);
        tiled_conv_auto(
            BATCH_SIZE, IN_DIM, IN_CHANNELS,
            OUT_CHANNELS, OUT_DIM,
            STRIDE, 1, KERNEL_DILATION, PADDING, KERNEL_DIM,
            false, false, false, false, false,

            (elem_t*)input,
            (elem_t*)weights_mat,
            NO_BIAS ? NULL : (acc_t*)bias,
            (elem_t*)output_mat,

            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, WS
        );
        uint64_t end_gemmini = read_cycles();
        printf("Gemmini auto conv took %llu cycles\n\n", end_gemmini - start_gemmini);

        gemmini_fence();

        start_gemmini = read_cycles();
        int retval = tiled_conv(
            BATCH_SIZE, IN_DIM, IN_CHANNELS,
            OUT_CHANNELS, OUT_DIM,
            STRIDE, 1, KERNEL_DILATION, PADDING, KERNEL_DIM,
            false, false, false, false, false,

            BATCHES,
            OROWS, OCOLS, OCHS,
            KROWS, KCOLS, KCHS,
            SPATIAL_OCHS, SPATIAL_KCHS,

            (elem_t*)input,
            (elem_t*)weights_mat,
            NO_BIAS ? NULL : (acc_t*)bias,
            (elem_t*)output_mat,

            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

            WS,
            PERM_STR);

        gemmini_fence();
        end_gemmini = read_cycles();
        if (retval != 0) {
            printf("Exit after %llu cycles\n\n", retval);
        } else {
            printf("Gemmini tiled conv took %llu cycles\n\n", end_gemmini - start_gemmini);
        }

    }
    return 0;
}
