// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define N 8

#if (N*DIM) > (BANK_NUM*BANK_ROWS)
#error not enough scratchpad space
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  gemmini_extended2_config_ld(DIM * sizeof(elem_t), MVIN_SCALE_IDENTITY, true);
  gemmini_config_ex(0, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0);
  gemmini_config_st(DIM * sizeof(elem_t));

  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);

  const uint32_t acc_addr = 1 << (ADDR_LEN-1);

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;

  for (size_t n = 0; n < N; ++n) {
    gemmini_mvin(In[n], acc_addr | (n*DIM));
    gemmini_mvout(Out[n], acc_addr | (n*DIM));
  }

  // printf("Fence");
  gemmini_fence();

  for (size_t n = 0; n < N; ++n)
    if (!is_equal(In[n], Out[n])) {
      printf("Matrix %u:\n", n);
      printMatrix(In[n]);
      printf("Matrix %u output:\n", n);
      printMatrix(Out[n]);
      printf("\n");

      exit(1);
    }

  exit(0);
}

