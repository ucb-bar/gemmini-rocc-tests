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

#define HEADS 1

#define ACTIVATION NO_ACTIVATION
// #define ACTIVATION SOFTMAX

#define NO_BIAS 0
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 1

#ifndef BAREMETAL

#define MAT_DIM_I 128 // 128 // 256
#define MAT_DIM_K 512 // 64 // 256
#define MAT_DIM_J 512 // 128 // 256

#else

// #define MAT_DIM_I 128
// #define MAT_DIM_K 128
// #define MAT_DIM_J 128

#define MAT_DIM_I 512 // 256
#define MAT_DIM_K 512 // 256
#define MAT_DIM_J 32 // 256

// #define MAT_DIM_I 256
// #define MAT_DIM_K 512
// #define MAT_DIM_J 512

#endif

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

    printf("HEADS: %d\n", HEADS);
    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);
    printf("ACTIVATION: %d\n", ACTIVATION);

    gemmini_flush(0);

#if A_TRANSPOSE==0
    static elem_t full_A[HEADS][MAT_DIM_I][MAT_DIM_K] row_align(1);
#else
    static elem_t full_A[HEADS][MAT_DIM_K][MAT_DIM_I] row_align(1);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[HEADS][MAT_DIM_K][MAT_DIM_J] row_align(1);
#else
    static elem_t full_B[HEADS][MAT_DIM_J][MAT_DIM_K] row_align(1);
#endif

    static elem_t full_C[HEADS][MAT_DIM_I][MAT_DIM_J] row_align(1);
    static acc_t full_D[HEADS][MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[HEADS][MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[HEADS][MAT_DIM_I][MAT_DIM_J];

    counter_configure(0, RDMA_BYTES_REC);
    counter_configure(1, WDMA_BYTES_SENT);
    counter_reset();

    printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);
    uint64_t start = read_cycles();

    for (int head = 0; head < HEADS; head++)
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            // (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            (elem_t*)full_A[head], (elem_t*)full_B[head], NO_BIAS ? NULL : &full_D[head][0][0], (elem_t*)full_C[head],
            // (elem_t*)full_A[0], (elem_t*)full_B[0], NO_BIAS ? NULL : &full_D[0][0][0], (elem_t*)full_C[0],
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);

    gemmini_fence();

    uint64_t end = read_cycles();
    printf("Cycles taken: %llu\n", end-start);

    const uint64_t total_macs = HEADS * MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);

    printf("RDMA_BYTES_REC: %u\n", counter_read(0));
    printf("WDMA_BYTES_SENT: %u\n", counter_read(1));

  exit(0);
}

