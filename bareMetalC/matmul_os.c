// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

int main() {
  for (int n = 0; n < 2; n++) {
    static matmul_t mm;

    for (size_t i = 0; i < DIM; ++i) {
      for (size_t j = 0; j < DIM; ++j) {
        // A = incrementing values row by row
        mm.A[i][j] = i*DIM + j;
        // B = identity matrix
        mm.B[i][j] = (i == j) ? 1 : 0;
        // D = zeros, TODO try to use matmul.preload(rd = 1) instead
        mm.D[i][j] = 0;
      }
    }

    init_matmul(&mm);

    printf("Moving in\n");
    for (size_t i = 0; i < DIM; ++i) {
      matmul_mvin(mm.A_tp[i], i);
      matmul_mvin(mm.B[i], DIM + i);
      matmul_mvin(mm.D[i], 2*DIM + i);
    }

    printf("Setting mode\n");
    matmul_setmode(0);
    printf("Preloading\n");
    matmul_preload_no_rd(2*DIM, 3*DIM);
    printf("Computing\n");
    matmul_compute_preloaded(0x0, DIM);

    printf("Moving out\n");
    for (size_t i = 0; i < DIM; ++i) {
      matmul_mvout(mm.C[i], 3*DIM + i);
    }

    printf("Gold:\n");
    printMatrix(mm.gold);
    printf("Actual:\n");
    printMatrix(mm.C);

    if (!matmul_is_correct(&mm))
        exit(1);
  }

  exit(0);
}

