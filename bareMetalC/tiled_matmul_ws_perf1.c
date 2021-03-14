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

#define NO_BIAS 0
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0
#define WARMUP 0

#ifndef BAREMETAL
#define MAT_DIM 512
#define MAT_DIM_I 256
#define MAT_DIM_K 512
#define MAT_DIM_J 256
#else
#define MAT_DIM 576
#define MAT_DIM_I MAT_DIM+64
#define MAT_DIM_K MAT_DIM_I
#define MAT_DIM_J MAT_DIM_I
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

#define SKIP_A false
#define SKIP_B true

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

#if A_TRANSPOSE==0
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#else
    static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(MAX_BLOCK_LEN);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
#else
    static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#endif

    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN);

	 printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM, MAT_DIM, MAT_DIM);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);

#if WARMUP == 1
	 unsigned long warm_start = read_cycles();

    tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE, SKIP_A, SKIP_B,
            WS);
	 unsigned long warm_end = read_cycles();
    printf("warm Cycles taken: %u\n", warm_end-warm_start);

    const int warm_total_macs = MAT_DIM * MAT_DIM * MAT_DIM;
    const int warm_ideal_cycles = warm_total_macs / (DIM * DIM);
    const int warm_utilization = 100 * warm_ideal_cycles / (warm_end-warm_start);
    printf("Utilization: %d%%\n", warm_utilization);
#endif
	 gemmini_flush(0);

	 unsigned long start = read_cycles();
    tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE, SKIP_A, SKIP_B,
            WS);
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    const int total_macs = MAT_DIM * MAT_DIM * MAT_DIM;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n", utilization);

  exit(0);
}

