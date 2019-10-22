// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "include/systolic.h"

int main() {
  static uint8_t A[DIM][DIM];
  static uint8_t A_out[DIM][DIM];
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      // A = incrementing values row by row
      A[i][j] = i*DIM + j;
    }
  }

  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvin(A[i], i);
  }

  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvout(A_out[i], i);
  }

  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      printf("A[%lu][%lu] = %d\n", i, j, A[i][j]);
      printf("A_out[%lu][%lu] = %d\n", i, j, A_out[i][j]);
      assert(A_out[i][j] == A[i][j]);
    }
  }

  return 0;
}
