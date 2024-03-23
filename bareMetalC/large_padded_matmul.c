// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1

#define DIM_I 20
#define DIM_K 45
#define DIM_J 60

// include helper functions for large matricies
#include "large_matmul_utils.h"

int main() {
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  }
#endif

  gemmini_flush(0);

  static elem_t full_A[DIM_I][DIM_K] row_align(1);
  static elem_t full_B[DIM_K][DIM_J] row_align(1);
  static elem_t full_C[DIM_I][DIM_J] row_align(1);

#if CHECK_RESULT == 1
  static int64_t gold_full[DIM_I][DIM_J];
  static elem_t gold[DIM_I][DIM_J];
#endif

  // printf("Init A\n");
  for (size_t i = 0; i < DIM_I; ++i) {
    for (size_t j = 0; j < DIM_K; ++j) {
      full_A[i][j] = rand() % 2;
    }
  }

  // printf("Init B\n");
  for (size_t i = 0; i < DIM_K; ++i) {
    for (size_t j = 0; j < DIM_J; ++j) {
      full_B[i][j] = rand() % 2;
    }
  }

#if CHECK_RESULT == 1
  printf("Starting slow CPU matmul\n");
  uint64_t cpu_start = read_cycles();
  full_matmul(full_A, full_B, gold_full);
  uint64_t cpu_end = read_cycles();
  printf("Cycles taken by CPU: %llu\n", cpu_end-cpu_start);
  full_matshift(gold_full, gold, 0);
#endif

  printf("Starting gemmini matmul\n");
  uint64_t start = read_cycles();

  tiled_matmul_auto(DIM_I, DIM_J, DIM_K,
                    full_A, full_B, NULL, full_C,
                    DIM_K, DIM_J, DIM_J, DIM_J, // striding factors
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, // mvin scaling factors
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, // activation, scale, relu6_shift, repeating_bias
                    false, false, // no transposing
                    false, false, // elem_t C, full bias width (doesn't matter since no bias added)
                    0,
                    WS);

  /* To run with hardcoded tiling factors, you can use this function instead:

  const size_t tile_I = 1;
  const size_t tile_J = 1;
  const size_t tile_K = 1;

  tiled_matmul(DIM_I, DIM_J, DIM_K,
               full_A, full_B, NULL, full_C,
               NO_ACTIVATION, 0, false, // activation, shift, repeating_bias
               tile_I, tile_J, tile_K,
               WS);
  */

  uint64_t end = read_cycles();
  printf("Cycles taken by Gemmini: %llu\n", end-start);

#if CHECK_RESULT == 1
  if (!full_is_equal(full_C, gold)) {
    printf("C:\n");
    full_printMatrix(full_C);
    printf("Gold:\n");
    full_printMatrix(gold);
    printf("\n");

    printf("FAIL\n");
    exit(1);
  }
#endif

  printf("PASS\n");
  exit(0);
}
