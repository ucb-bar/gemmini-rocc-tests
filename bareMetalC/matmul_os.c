// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

void matmul(elem_t A[DIM][DIM], elem_t B[DIM][DIM], elem_t D[DIM][DIM], elem_t C[DIM][DIM]) {
    for (size_t r = 0; r < DIM; r++)
        for (size_t c = 0; c < DIM; c++) {
            C[r][c] = D[r][c];
            for (size_t k = 0; k < DIM; k++)
                C[r][c] += A[r][k]*B[k][c];
        }
}

void transpose(elem_t in[DIM][DIM], elem_t out[DIM][DIM]) {
    for (size_t r = 0; r < DIM; r++)
        for (size_t c = 0; c < DIM; c++)
            out[c][r] = in[r][c];
}

void printMatrix(elem_t m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int main() {
  static elem_t A[DIM][DIM];
  static elem_t B[DIM][DIM];
  static elem_t C[DIM][DIM];
  static elem_t D[DIM][DIM];

  static elem_t gold[DIM][DIM];

  static elem_t A_tp[DIM][DIM];

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

  matmul(A, B, D, gold);
  transpose(A, A_tp);

#ifdef DEBUG_PRINTS
  printMatrix(A);
  printMatrix(B);
#endif
  printf("Moving in\n");

  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvin(A_tp[i], i);
    matmul_mvin(B[i], DIM + i);
    matmul_mvin(D[i], 2*DIM + i);
  }

  printf("Setting mode\n");
  matmul_setmode(0);
  printf("Preloading\n");
  matmul_preload_no_rd(2*DIM, 3*DIM);
  printf("Computing\n");
  matmul_compute_preloaded(0x0, DIM);

  printf("Moving out\n");
  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvout(C[i], 3*DIM + i);
  }

  printf("Gold:\n");
  printMatrix(gold);
  printf("Actual:\n");
  printMatrix(C);

  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      if (C[i][j] != gold[i][j]) {
        printf("C[%lu][%lu] = %d\n", i, j, C[i][j]);
        printf("Gold[%lu][%lu] = %d\n", i, j, gold[i][j]);
        exit(1);
      }
    }
  }
  exit(0);
}

