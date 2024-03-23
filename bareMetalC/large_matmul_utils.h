// See LICENSE for license details.

#ifndef LARGE_MATMUL_UTILS_H
#define LARGE_MATMUL_UTILS_H

#include <stdio.h>
#include "include/gemmini_params.h"

// similar helper functions to gemmini_testutils.h but larger

void lmm_util_print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in + r*DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[DIM_I][DIM_K], elem_t B[DIM_K][DIM_J], full_t C_full[DIM_I][DIM_J]) {
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

void full_matshift(full_t full[DIM_I][DIM_J], elem_t out[DIM_I][DIM_J], int shift) {
  for (size_t r = 0; r < DIM_I; r++)
    for (size_t c = 0; c < DIM_J; c++) {
      // Bitshift and round element
      full_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
      full_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
    }
}

#endif // LARGE_MATMUL_UTILS_H
