// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini_testutils.h"

elem_t rshift(elem_t x, int shift) {
  return shift >= 0 ? ROUNDING_RIGHT_SHIFT(x, shift) : x << -shift;
}

int main() {
  elem_t A[DIM][DIM];
  elem_t B[DIM][DIM];
  elem_t C[DIM][DIM];
  elem_t gold[DIM][DIM];

  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      A[i][j] = (rand() % 16) - 8;
      B[i][j] = (rand() % 16) - 8;
    }

  for (int ashift = -2; ashift < 2; ashift++) {
    for (int bshift = -2; bshift < 2; bshift++) {
      for (size_t i = 0; i < DIM; i++)
        for (size_t j = 0; j < DIM; j++) {
          acc_t sum = rshift(A[i][j], ashift) + rshift(B[i][j], bshift);
          gold[i][j] = sum > elem_t_max ? elem_t_max :
            (sum < elem_t_min ? elem_t_min : sum);
        }

      uint32_t A_acc_addr = 1 << (ADDR_LEN - 1);
      uint32_t B_acc_addr = (1 << (ADDR_LEN - 1)) | (1 << (ADDR_LEN - 2));
      uint32_t C_acc_addr = 1 << (ADDR_LEN - 1);

      gemmini_extended2_config_ld(DIM * sizeof(elem_t), ashift, true);
      gemmini_mvin(A, A_acc_addr);

      gemmini_extended2_config_ld(DIM * sizeof(elem_t), bshift, true);
      gemmini_mvin(B, B_acc_addr);

      gemmini_config_ex(0, NO_ACTIVATION, 0, 0, 0);
      gemmini_mvout(C, C_acc_addr);

      gemmini_fence();

      if (!is_equal(C, gold)) {
        printf("Wrong (ashift: %d, bshift: %d)\n", ashift, bshift);
        printf("\"C\" matrix:\n");
        printMatrix(C);
        printf("\n");
        printf("\"Gold\" matrix:\n");
        printMatrix(gold);
        printf("\n");
        printf("\"A\" matrix:\n");
        printMatrix(A);
        printf("\n");
        printf("\"B\" matrix:\n");
        printMatrix(B);
        printf("\n");
        printf("Wrong (ashift: %d, bshift: %d)\n", ashift, bshift);
        exit(1);
      }
    }
  }

  exit(0);
}

