// See LICENSE for license details.

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

#include "data_resadd.h"
#define NUM_INT 4
#define NUM_FP 2
#define USE_RELU true

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

    printf("I: %d, J: %d\n", MAT_DIM_I, MAT_DIM_J);

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

    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    printf("Starting gemmini resadd\n");
    unsigned long start = read_cycles();
    tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, (elem_t*)A, (elem_t*)B,
            (elem_t*)C, USE_RELU, WS);
    rr_fence(cfgid);
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
    rr_release(cfgid);

    if (!full_is_equal(C, gold)) {
      printf("C:\n");
      full_printMatrix(C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("A:\n");
      full_printMatrix(A);
      printf("B:\n");
      full_printMatrix(B);
      printf("\n");

      exit(1);
    }
  exit(0);
}

