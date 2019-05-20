// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"

#define N 8

#if (N*DIM) > (BANK_NUM*BANK_ROWS)
#error not enough scratchpad space
#endif

int main() {
  static elem_t In[N][DIM][DIM] row_align;
  static elem_t Out[N][DIM][DIM] row_align;

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;

  for (size_t n = 0; n < N; ++n) {
    matmul_mvin(In[n], n*DIM, 1, 0, 0, 0);
    matmul_mvout(Out[n], n*DIM, 0, 1, 0, 0);
  }

  matmul_fence();

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

