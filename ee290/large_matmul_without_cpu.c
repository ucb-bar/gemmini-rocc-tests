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

#define CHECK_RESULT 0

#define DIM_I 64
#define DIM_K 64
#define DIM_J 64

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], int64_t C_full[DIM_I][DIM_J]) {
  for (size_t r = 0; r < DIM_I; r++)
    for (size_t c = 0; c < DIM_J; c++) {
      C_full[r][c] = 0;
      for (size_t k = 0; k < DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t m[DIM_I][DIM_J]) {
  for (size_t i = 0; i < DIM_I; ++i) {
    for (size_t j = 0; j < DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[DIM_I][DIM_J], elem_t y[DIM_I][DIM_J]) {
  for (size_t i = 0; i < DIM_I; ++i)
    for (size_t j = 0; j < DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matshift(int64_t full[DIM_I][DIM_J], elem_t out[DIM_I][DIM_J], int shift) {
  for (size_t r = 0; r < DIM_I; r++)                             
    for (size_t c = 0; c < DIM_J; c++) {
      // Bitshift and round element
      int64_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
      int64_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
    }
} 

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

  static int64_t gold_full[DIM_I][DIM_J];
  static elem_t gold[DIM_I][DIM_J];

#if CHECK_RESULT == 1
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
#endif

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
                    NO_ACTIVATION, 0, false, // activation, shift, repeating_bias
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

