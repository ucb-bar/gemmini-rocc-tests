// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// #define PRINT_TILE %PRINT_TILE%
#define NO_BIAS 1
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

// #define NUM_LAYERS %NUM_LAYERS%
#define PE_DIM %PE_DIM%

// //in Gemmini, K is shared dimension
// #define MAT_DIM_I_LIST "%DIM_I%"
// #define MAT_DIM_K_LIST "%DIM_K%"
// #define MAT_DIM_J_LIST "%DIM_J%"
// #define TILE_I_LIST "%TILE_DIM_I%"
// #define TILE_K_LIST "%TILE_DIM_K%"
// #define TILE_J_LIST "%TILE_DIM_J%"

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    // static elem_t full_A[10000][10000] row_align(1);
    // static elem_t full_B[10000][10000] row_align(1);
    // static elem_t full_C[10000][10000] row_align(1);
    // static acc_t full_D[10000][10000] row_align_acc(1);

    static elem_t full_A[128][128][2] row_align(1);
    static elem_t full_B[128][128][2] row_align(1);
    static elem_t full_C[128][128][2] row_align(1);
    static acc_t full_D[1][1] row_align_acc(1);
    // static full_t gold_full[1024][1024];
    // static elem_t gold[1024][1024];

    const int NUM_LAYERS = %NUM_LAYERS%;
    const int MAT_DIM_I_LIST[%NUM_LAYERS%] = {%DIM_I%};
    const int MAT_DIM_K_LIST[%NUM_LAYERS%] = {%DIM_K%};
    const int MAT_DIM_J_LIST[%NUM_LAYERS%] = {%DIM_J%};
    const int TILE_I_LIST[%NUM_LAYERS%] = {%TILE_OCOLS%};
    const int TILE_K_LIST[%NUM_LAYERS%] = {%TILE_KCHS%};
    const int TILE_J_LIST[%NUM_LAYERS%] = {%TILE_OCHS%};
    const int SPATIAL_TILE_K_LIST[%NUM_LAYERS%] = {%SPATIAL_TILE_KCHS%};
    const int SPATIAL_TILE_J_LIST[%NUM_LAYERS%] = {%SPATIAL_TILE_OCHS%};
    char *PERM_STR_LIST[%NUM_LAYERS%] = {%PERM_STR%};

    printf("Starting gemmini matmul\n");
    // printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    // printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);

    for (int l = 0; l < NUM_LAYERS; l++) {
        const int MAT_DIM_I = MAT_DIM_I_LIST[l];
        const int MAT_DIM_K = MAT_DIM_K_LIST[l];
        const int MAT_DIM_J = MAT_DIM_J_LIST[l];
        int TILE_I = TILE_I_LIST[l];
        int TILE_K = TILE_K_LIST[l];
        int TILE_J = TILE_J_LIST[l];
        const int SPATIAL_TILE_K = SPATIAL_TILE_K_LIST[l];
        const int SPATIAL_TILE_J = SPATIAL_TILE_J_LIST[l];
        char *PERM_STR = PERM_STR_LIST[l];

        // printf("I: %d\n, K: %d\n, J: %d\n, TILE_I: %d\n, TILE_K: %d\n, TILE_J: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K, TILE_I, TILE_K, TILE_J);
        // printf("%d_%d_%d_%d_%d_%d_%d_%d_%s\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K, TILE_I, TILE_K, TILE_J, SPATIAL_TILE_K, SPATIAL_TILE_J, PERM_STR);

        // tiled_matmul_auto_inner(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
        //         (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
        //         A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
        //         MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        //         NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
        //         A_TRANSPOSE, B_TRANSPOSE,
        //         false, false,
        //         0,
        //         WS,
        //         true);

        // gemmini_flush(0);

        // unsigned long start = read_cycles();

        // tiled_matmul_auto_inner(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
        //         (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
        //         A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
        //         MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        //         NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
        //         A_TRANSPOSE, B_TRANSPOSE,
        //         false, false,
        //         0,
        //         WS,
        //         false);

        // gemmini_fence();

        // unsigned long end = read_cycles();
        // printf("Gemmini auto matmul took %llu cycles\n\n", end - start);

        // TILE_I = (TILE_I < 16) ? 1 : TILE_I >> 4;
        // TILE_J = (TILE_J < 16) ? 1 : TILE_J >> 4;
        // TILE_K = (TILE_K < 16) ? 1 : TILE_K >> 4;

        gemmini_fence();
        unsigned long tiled_start = read_cycles();

        uint64_t retval = tiled_matmul(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
                A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                TILE_I, TILE_J, TILE_K,
                A_TRANSPOSE, B_TRANSPOSE,
                false, false,
                0,
                WS,
                PERM_STR);

        gemmini_fence();

        unsigned long tiled_end = read_cycles();
        if (retval != 0) {
            printf("Exit after %llu cycles\n\n", retval);
        } else {
            printf("Gemmini tiled matmul took %llu cycles\n\n", tiled_end - tiled_start);
        }

        // const int total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
        // const int ideal_cycles = total_macs / (DIM * DIM);
        // const int utilization = 100 * ideal_cycles / (end-start);
        // printf("Utilization: %d%%\n", utilization);
    }
    exit(0);
}

