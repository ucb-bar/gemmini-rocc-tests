// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

#define N (8)

int main() {
  static elem_t In[N][DIM][DIM];
  static elem_t Out[N][DIM][DIM];

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;

  for (size_t n = 0; n < N; ++n) {
    matmul_mvin(In[n], n*DIM, 1, 0, 0, 0);
    matmul_mvout(Out[n], n*DIM, 0, 1, 0, 0);
  }

  matmul_fence();

  // for (size_t n = 0; n < N; ++n) {
  //   printf("Matrix %u:\n", n);
  //   printMatrix(In[n]);
  //   printf("Matrix %u output:\n", n);
  //   printMatrix(Out[n]);
  //   printf("\n");
  // }

  for (size_t n = 0; n < N; ++n)
    if (!is_equal(In[n], Out[n]))
      exit(1);

  exit(0);
}

