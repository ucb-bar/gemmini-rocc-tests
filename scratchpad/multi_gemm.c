// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define FLOAT false
#include "include/gemmini_testutils.h"
#include "include/rerocc.h"

#define ACTIVATION NO_ACTIVATION
#define BASE_ADDR 0x70000000L

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1
#define PageSize 4096

#define CHECK_RESULT 1
//#define ACC_ID 3
#define ACC_T acc_t

#include "data_matmul.h"
#define NUM_INT 8
#define NUM_FP 5

#define NUM_ARRAY 8
//#define REPEATING_BIAS false

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}
int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

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
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);

    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));
    printf("A copy addr: 0x%08lx\n", A_copy_addr);
    uint64_t B_copy_addr = A_copy_addr + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t);// + 64*3;
    uint64_t C_copy_addr = B_copy_addr + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t);// + 64*3;
    uint64_t D_copy_addr = C_copy_addr + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t);
/*
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 64) - 32;
        //B[i][j] = (rand() % 8) - 4;
        //gold[i][j] = A[i][j]+B[i][j];
      }
    }
    */
    printf("perform memcpy\n");
    bool granted = false;
    int index = 0;
    printf("copy A\n");
    memcpy((elem_t*) A_copy_addr, (elem_t*) full_A, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_K);
    printf("copy B\n");
    memcpy((elem_t*) B_copy_addr, (elem_t*) full_B, sizeof(elem_t)*MAT_DIM_K*MAT_DIM_J);
    printf("copy D\n");
    if(!NO_BIAS) memcpy((acc_t*) D_copy_addr, (acc_t*) full_D, REPEATING_BIAS ? sizeof(acc_t) * MAT_DIM_J : sizeof(acc_t)*MAT_DIM_I*MAT_DIM_J);

    printf("gemmini spad rows: %d, acc rows: %d \n", BANK_ROWS, ACC_ROWS);
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);

    printf("Starting gemmini matmul\n");
    uint64_t start = read_cycles();

    multi_tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
        (elem_t*) A_copy_addr, (elem_t*) B_copy_addr, NO_BIAS ? NULL : (acc_t*) D_copy_addr, (elem_t*) C_copy_addr,
        MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
        false, false,
        false, false,
        0,
        NUM_ARRAY);

    uint64_t end = read_cycles();
    printf("Cycles taken: %llu\n", end-start);

    const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);
    for(int i = 0; i < NUM_ARRAY; i++)
      rr_release(i);
    printf("copy C\n");
    memcpy((elem_t*) C, (elem_t*) C_copy_addr, sizeof(elem_t)*MAT_DIM_I*MAT_DIM_J);
#if CHECK_RESULT == 1
    if (!full_is_equal(C, gold)) {
      printf("C:\n");
      full_printMatrix(C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
#endif
    exit(0);
}

