// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

int main() {
  const int d_additions = 10;

  static elem_t IDENTITY[DIM][DIM];
  static elem_t result[DIM][DIM];
  static elem_t gold[DIM][DIM];

  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      IDENTITY[i][j] = i == j;
      gold[i][j] = i == j ? (d_additions+1) : 0;
    }
  }

  int A_addr = 0;
  int B_addr = BANK_SIZE;
  int D_addr = 2*BANK_SIZE;

  // printf("Moving in\n");
  matmul_mvin(IDENTITY, A_addr, 0, 0, 0, 0);
  matmul_mvin(IDENTITY, B_addr, 0, 0, 0, 0);
  matmul_mvin(IDENTITY, D_addr, 0, 0, 1, 0);
  
  // printf("Setting mode\n");
  matmul_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 1, 0, 0);

  // printf("RAW with D addr\n");
  for (int i = 0; i < d_additions; i++) {
    if (i == d_additions-1) {
      matmul_preload(D_addr, D_addr, 0, 0, 1, 0);
    } else {
      matmul_preload(D_addr, D_addr, 0, 0, 0, 0);
    }
    matmul_compute_preloaded(A_addr, B_addr);
  }

  // printf("Moving out\n");
  matmul_mvout(result, D_addr, 0, 0, 0, 1);

  matmul_fence();

  /*printMatrix(result);
  printf("\n");
  printMatrix(gold);*/

  if (!is_equal(result, gold))
    exit(1);

  exit(0);
}

