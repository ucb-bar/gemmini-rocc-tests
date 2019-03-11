// See LICENSE for license details.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "include/systolic.h"

#define DIM 16

int main() {
  // TODO: Should be signed, but then need to add a bias term below
  static uint8_t A[DIM][DIM];
  static uint8_t B[DIM][DIM];
  static uint8_t C[DIM][DIM];
  static uint8_t D[DIM][DIM];
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      // A = incrementing values row by row
      A[i][j] = i*DIM + j;
      // B = identity matrix
      B[i][j] = (i == j) ? 1 : 0;
      // D = zeros, TODO try to use matmul.preload(rd = 1) instead
      D[i][j] = 0;
    }
  }

#ifdef DEBUG_PRINTS
  for (size_t j = 0; j < DIM; ++j) {
    printf("i = %lu, j = %lu, A_ij = %d\n", i, j, A[i][j]);
    printf("i = %lu, j = %lu, B_ij = %d\n", i, j, B[i][j]);
  }
#endif


  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvin(A[i], i);
    matmul_mvin(B[i], DIM + i);
    matmul_mvin(D[i], 2*DIM + i);
  }

  matmul_setmode(0);
  matmul_preload_no_rd(3*DIM, 2*DIM);
  matmul_compute_preloaded(0x0, DIM);

  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvout(C[i], 3*DIM + i);
  }
#define DEBUG_PRINTS
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
#ifdef DEBUG_PRINTS
      printf("C[%lu][%lu] = %d\n", i, j, C[i][j]);
      printf("A[%lu][%lu] = %d\n", i, j, A[i][j]);
#endif
      assert(C[i][j] == A[i][j]);
    }
  }

  return 0;
}
