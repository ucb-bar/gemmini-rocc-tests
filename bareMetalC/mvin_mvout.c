// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

int is_equal(elem_t X[DIM][DIM], elem_t Y[DIM][DIM]) {
    for (size_t i = 0; i < DIM; ++i) {
        for (size_t j = 0; j < DIM; ++j) {
            if (X[i][j] != Y[i][j]) {
                printf("X[%lu][%lu] = %u\n", i, j, X[i][j]);
                printf("X_out[%lu][%lu] = %u\n", i, j, Y[i][j]);
                if (X[i][j] != Y[i][j]) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

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
      D[i][j] = i*DIM + j + 2;
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

  if (!is_equal(A, A_out) || !is_equal(B, B_out)
          || !is_equal(C, C_out) || !is_equal(D, D_out)) {
      exit(1);
  }

  exit(0);
}
