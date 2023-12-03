// See LICENSE for license details.
// skip mvin/mvout, only execution

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT false
#include "include/gemmini_testutils.h"


#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_K 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 256
#define MAT_DIM_K 512
#define MAT_DIM_J (512*8)
#endif

#define NUM_INT 8
#define NUM_FP 5

#define NUM_ARRAY 8

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    int cfgid = 0;
    for(int i = 0; i < NUM_INT + NUM_FP; i++){   
#if FLOAT
        if(i < NUM_INT)
            continue;
#else
        if(i >= NUM_INT)
            continue;
#endif
        bool acquired = rr_acquire_single(cfgid, i);
        if(acquired){
            printf("gemmini %d acquired to cfgid %d\n", i, cfgid);
            cfgid ++;
            if(cfgid == NUM_ARRAY)
                break;
        }
    }
    for(int i = 0; i < NUM_ARRAY; i++){
      rr_set_opc(XCUSTOM_ACC, i);
      gemmini_flush(0);
    }

    printf("Starting gemmini matmul\n");
    unsigned long start = read_cycles();

    multi_tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            NULL, NULL, NULL, NULL,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            NUM_ARRAY);

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM * NUM_ARRAY);
    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);


  exit(0);
}

