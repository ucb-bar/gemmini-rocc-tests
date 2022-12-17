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

#define NO_BIAS 0
#define FULL_BIAS_WIDTH 1

//#define APPROX

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#ifndef BAREMETAL

#define MAT_DIM_N 240

#else
//#define MAT_DIM_N 30
#define MAT_DIM_N 64

#endif

void full_printMatrix(elem_t m[MAT_DIM_N][MAT_DIM_N]) {
  for (size_t i = 0; i < MAT_DIM_N; ++i) {
    for (size_t j = 0; j < MAT_DIM_N; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_N][MAT_DIM_N], elem_t y[MAT_DIM_N][MAT_DIM_N]) {
  for (size_t i = 0; i < MAT_DIM_N; ++i)
    for (size_t j = 0; j < MAT_DIM_N; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

int main() {
#if defined(FAST) || !defined(HAS_NORMALIZATIONS)
    exit(0);
#endif

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("MAT_DIM_N: %d\n", MAT_DIM_N);
    printf("NO_BIAS: %d\n", NO_BIAS);

    gemmini_flush(0);

    static elem_t A[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t B[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t C[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t D[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t t0[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t t1[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t t0_gold[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static elem_t t1_gold[MAT_DIM_N][MAT_DIM_N] row_align(1);
    static ACC_T bias0[MAT_DIM_N][MAT_DIM_N] row_align_acc(1);
    static ACC_T bias1[MAT_DIM_N][MAT_DIM_N] row_align_acc(1);
    static ACC_T bias2[MAT_DIM_N][MAT_DIM_N] row_align_acc(1);

    static elem_t gold[MAT_DIM_N][MAT_DIM_N];
    static elem_t res[MAT_DIM_N][MAT_DIM_N];

    acc_scale_t scale0 = 2.5; // normal matmul
    acc_scale_t scale1 = 10.0; // layer norm
    acc_scale_t scale2 = 1.0; // softmax

#if CHECK_RESULT == 1
    printf("Initializing matrices\n");
    for (size_t i = 0; i < MAT_DIM_N; ++i) {
      for (size_t j = 0; j < MAT_DIM_N; ++j) {
        A[i][j] = (rand() % 15) - 7;
        B[i][j] = (rand() % 15) - 7;
        C[i][j] = (rand() % 15) - 7;
        D[i][j] = (rand() % 15) - 7;
      }
    }

    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();

//     gold/res = scale2 * sm(scale1 * (ln((scale0 * (A * B + bias0)) * C + bias1)) * D + bias2)

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &bias0[0][0], (elem_t*)t0_gold,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, scale0, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            CPU);

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)t0_gold, (elem_t*)C, NO_BIAS ? NULL : &bias1[0][0], (elem_t*)t1_gold,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            LAYERNORM, scale1, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            CPU);

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)t1_gold, (elem_t*)D, NO_BIAS ? NULL : &bias2[0][0], (elem_t*)gold,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, scale2, 0.05, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            CPU);

    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);

#endif

    printf("Starting gemmini matmul\n");
    unsigned long start = read_cycles();

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &bias0[0][0], (elem_t*)t0,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, scale0, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)t0, (elem_t*)C, NO_BIAS ? NULL : &bias1[0][0], (elem_t*)t1,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            LAYERNORM, scale1, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);
    gemmini_fence();

    tiled_matmul_auto(MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            (elem_t*)t1, (elem_t*)D, NO_BIAS ? NULL : &bias2[0][0], (elem_t*)res,
            MAT_DIM_N, MAT_DIM_N, MAT_DIM_N, MAT_DIM_N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, scale2, 0.05, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    gemmini_fence();

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

#if CHECK_RESULT == 1
    if (!full_is_equal(res, gold)) {
      printf("Mismatch\n");
//      if (!full_is_equal(t0, t0_gold)) {
//        printf("t0\n");
//        full_printMatrix(t0);
//        printf("t0_gold\n");
//        full_printMatrix(t0_gold);
//      }
//      if (!full_is_equal(t1, t1_gold)) {
//        printf("t1\n");
//        full_printMatrix(t1);
//        printf("t1_gold\n");
//        full_printMatrix(t1_gold);
//      }
      printf("Result:\n");
      full_printMatrix(res);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

