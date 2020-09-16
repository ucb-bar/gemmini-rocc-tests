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

#define CHECK_RESULT 1

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512
#else
#define MAT_DIM_K 32
#define MAT_DIM_I 32
#define MAT_DIM_J 32
#endif

#define R_SHIFT 0
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

    gemmini_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t R[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
    static acc_t D[MAT_DIM_I][MAT_DIM_J];
    static elem_t out[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
    // printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        R[i][j] = (rand() % 8) - 4;
	D[i][j] = 2;
      }
    }
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        A[i][j] = (rand() % 16) - 8;
      }
    }
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        B[i][j] = (rand() % 16) - 8;
      }
    }

printf("normal matmul \n");
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
		    (elem_t*)A, (elem_t*)B, (acc_t*)D, (elem_t*)C,
		    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    NO_ACTIVATION, 2, 0, true,
                    WS);
printf("resadd \n");
    tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, R_SHIFT, (elem_t*)R, (elem_t*)C,
            (elem_t*)gold, USE_RELU, WS);
 
printf("matmul + resadd \n");
    tiled_matmul_resadd_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
		    (elem_t*)A, (elem_t*)B, (acc_t*)D, (elem_t*)out,
		    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
                    NO_ACTIVATION, 2, 0, true,
                    WS, (elem_t*)R, R_SHIFT, USE_RELU);
/*

    printf("Starting slow CPU resadd\n");
    unsigned long cpu_start = read_cycles();
    resadd_cpu(MAT_DIM_I, MAT_DIM_J, A_SHIFT, (elem_t*)A, (elem_t*)B,
            (elem_t*)gold, USE_RELU);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
*/
#endif
/*
    printf("Starting gemmini resadd\n");
    unsigned long start = read_cycles();
    tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, A_SHIFT, (elem_t*)A, (elem_t*)B,
            (elem_t*)C, USE_RELU, WS);
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
*/
#if CHECK_RESULT == 1
    if (!full_is_equal(out, gold)) {
      printf("C:\n");
      full_printMatrix(C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("out:\n");
      full_printMatrix(out);
      printf("R:\n");
      full_printMatrix(R);
      printf("\n");
      printf("wrong \n");
      exit(1);
    }
#endif
      printf("C:\n");
      full_printMatrix(C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("out:\n");
      full_printMatrix(out);
      printf("R:\n");
      full_printMatrix(R);
      printf("\n");
      printf("correct \n");

  exit(0);
}

