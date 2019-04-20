// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

#define BIG_DIM 16

int is_equal_big(elem_t x[BIG_DIM][BIG_DIM], elem_t y[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i)
    for (size_t j = 0; j < BIG_DIM; ++j)
      if (x[i][j] != y[i][j])
          return 0;
  return 1;
}

void printMatrix_big(elem_t m[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i) {
    for (size_t j = 0; j < BIG_DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int main() {
  printf("Total rows required: %d\n", BIG_DIM*BIG_DIM/DIM);
  if (BIG_DIM % DIM != 0) {
    printf("Incorrect dimensions\n");
    exit(1);
  }

  static elem_t In[BIG_DIM][BIG_DIM];
  static elem_t Out[BIG_DIM][BIG_DIM];

  for (size_t i = 0; i < BIG_DIM; ++i)
    for (size_t j = 0; j < BIG_DIM; ++j)
      In[i][j] = i*BIG_DIM + j;

  matmul_config_ld(BIG_DIM*sizeof(elem_t), 0, 0, 0, 0);
  matmul_config_st(BIG_DIM*sizeof(elem_t), 0, 0, 0, 0);

  for (size_t i = 0; i < BIG_DIM; i += DIM) {
    for (size_t j = 0; j < BIG_DIM; j += DIM) {
      elem_t * dram_addr_in = &In[i][j];
      elem_t * dram_addr_out = &Out[i][j];
      int sp_addr = i*(BIG_DIM/DIM) + j;

      matmul_mvin(dram_addr_in, sp_addr, 1, 0, 0, 0);
      matmul_mvout(dram_addr_out, sp_addr, 0, 1, 0, 0);
    }
  }

  matmul_fence();

  // printf("Matrix:\n");
  // printMatrix_big(In);
  // printf("Matrix output:\n");
  // printMatrix_big(Out);
  // printf("\n");

  if (!is_equal_big(In, Out))
    exit(1);

  exit(0);
}

