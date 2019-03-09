// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include "include/systolic.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "util.h"

#define DIM 16

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
    matmul_mvin(A[i], i*DIM*sizeof(uint8_t));
  }

  for (size_t i = 0; i < DIM; ++i) {
    matmul_mvout(A_out[i], i*DIM*sizeof(uint8_t));
  }

  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      if (A_out[i][j] != A[i][j]) {
        printf("A[%lu][%lu] = %d\n", i, j, A[i][j]);
        printf("A_out[%lu][%lu] = %d\n", i, j, A_out[i][j]);
        exit(1);
      }
    }
  }
  exit(0);
}
