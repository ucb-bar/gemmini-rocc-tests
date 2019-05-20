// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/systolic.h"
#include "util.h"

#define BIG_DIM 64

#if (BIG_DIM % DIM) != 0
#error incorrect dimensions
#endif

#if (BIG_DIM * BIG_DIM / DIM) > (BANK_ROWS * BANK_NUM)
#error not enough rows
#endif

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
  for (int len = 1; len <= BIG_DIM/DIM; len++) {
    // printf("len: %d\n", len);

    static elem_t In[BIG_DIM][BIG_DIM] row_align;
    static elem_t Out[BIG_DIM][BIG_DIM] row_align;

    for (size_t i = 0; i < BIG_DIM; ++i)
      for (size_t j = 0; j < BIG_DIM; ++j)
        In[i][j] = i*BIG_DIM + j;

    matmul_config_ld(BIG_DIM*sizeof(elem_t), 0, 0, 0, 0);
    matmul_config_st(BIG_DIM*sizeof(elem_t), 0, 0, 0, 0);

    for (size_t i = 0; i < BIG_DIM; i += DIM) {
      for (size_t j = 0; j < BIG_DIM; j += DIM) {
        // printf("i: %u, j: %u\n", i, j);

        elem_t * dram_addr_in = &In[i][j];
        elem_t * dram_addr_out = &Out[i][j];
        int sp_addr = i*(BIG_DIM/DIM) + j;

        int already_moved_in = (j/DIM) % len != 0;

        if (!already_moved_in) {
          matmul_block_mvin(dram_addr_in, sp_addr, len, 1, 0, 0, 0);
          matmul_mvout(dram_addr_out, sp_addr, 0, 1, 0, 0);
        } else {
          matmul_mvout(dram_addr_out, sp_addr, 0, 0, 0, 0);
        }
      }
    }

    matmul_fence();

    if (!is_equal_big(In, Out)) {
      printf("len: %d\n", len);

      printf("Matrix:\n");
      printMatrix_big(In);
      printf("Matrix output:\n");
      printMatrix_big(Out);
      printf("\n");

      exit(1);
    }
  }

  exit(0);
}

