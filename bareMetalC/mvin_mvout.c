// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

int main() {
  static elem_t A[DIM][DIM];
  static elem_t A_out[DIM][DIM];
  static elem_t B[DIM][DIM];
  static elem_t B_out[DIM][DIM];
  static elem_t C[DIM][DIM];
  static elem_t C_out[DIM][DIM];
  static elem_t D[DIM][DIM];
  static elem_t D_out[DIM][DIM];

  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      A[i][j] = i*DIM + j;
      B[i][j] = i*DIM + j + 1;
      C[i][j] = i*DIM + j + 2;
      D[i][j] = i*DIM + j + 3;
    }
  }

  printf("Moving in\n");
  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvin(A[i], i);
    matmul_mvin(B[i], DIM + i);
    matmul_mvin(C[i], 2*DIM + i);
    matmul_mvin(D[i], 3*DIM + i);
  }

  printf("Moving out\n");
  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvout(A_out[i], i);
    matmul_mvout(B_out[i], DIM+i);
    matmul_mvout(C_out[i], 2*DIM+i);
    matmul_mvout(D_out[i], 3*DIM+i);
  }
  
  printf("A:\n");
  printMatrix(A);
  printf("A_out:\n");
  printMatrix(A_out);
  printf("B:\n");
  printMatrix(B);
  printf("B_out:\n");
  printMatrix(B_out);
  printf("C:\n");
  printMatrix(C);
  printf("C_out:\n");
  printMatrix(C_out);
  printf("D:\n");
  printMatrix(D);
  printf("D_out:\n");
  printMatrix(D_out);

  if (!is_equal(A, A_out) || !is_equal(B, B_out)
          || !is_equal(C, C_out) || !is_equal(D, D_out)) {
      exit(1);
  }

  exit(0);
}
