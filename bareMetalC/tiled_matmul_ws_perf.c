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

#define ACTIVATION NO_ACTIVATION

#define NO_BIAS 0
#define REPEATING_BIAS 0

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_K 512
#define MAT_DIM_J 256

#else

#define MAT_DIM_I 128
#define MAT_DIM_K 128
#define MAT_DIM_J 128

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

    int cfgid = 0;
    int i = 0;
    //for(int i = 0; i < 2; i++){
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            //break;
        }
    //}
    rr_set_opc(XCUSTOM_ACC, cfgid);
    gemmini_flush(0);

#if A_TRANSPOSE==0
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
#else
    static elem_t full_A[MAT_DIM_K][MAT_DIM_I] row_align(1);
#endif

#if B_TRANSPOSE==0
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
#else
    static elem_t full_B[MAT_DIM_J][MAT_DIM_K] row_align(1);
#endif

    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static acc_t full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];


    for(int i = 0 ; i < 2; i++){
    printf("Starting gemmini matmul\n");
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);
    uint64_t start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            false, false,
            0,
            WS);

    gemmini_fence();
    rr_fence(cfgid);

    uint64_t end = read_cycles();
    printf("Cycles taken: %llu\n", end-start);

    const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);
    }
    rr_release(cfgid);

  exit(0);
}

