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
#define SPARSITY 10

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

  // printf("Flush\n");
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  // create
  static elem_t InDense[DIM][DIM] row_align(1);
  static elem_t OutDense[DIM][DIM] row_align(1);

  static elem_t InSpData[DIM*DIM];    // it can be smaller, but I'm maxing it for now, the extra bits should be ignored
  static elem_t InSpCoo[DIM*DIM][2];  // it can be smaller, but I'm maxing it for now, the extra bits should be ignored
  
  // pre-populate
  unsigned int sparse_index = 0;
  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j)
      if ((rand() % 100) < SPARSITY) {
        int value = i*DIM + j + n;
        InDense[i][j]          = value;
        InSpData[sparse_index] = value;
        InSpCoo[sparse_index][0] = value;
        InSpCoo[sparse_index][1] = value;
      }

  // Apply mvin mvout
  gemmini_extended_mvin_sparse_coo(&InSpData[0], &InSpCoo[0], 0, 0, DIM, 0, DIM); // I need a catch for when the last entries are 0...
  gemmini_mvout(Out[n], n*DIM);
  //for (size_t n = 0; n < N; ++n) {
  //  // printf("Mvin %d\n", n);
  //  gemmini_extended_mvin_sparse_coo(&InSpData[0], &InSpCoo[0], spad_addr, start_col, cols, start_row, rows) \
  //  gemmini_mvin_sparse_coo(In[n], n*DIM);
  //  // printf("Mvout %d\n", n);
  //  gemmini_mvout(Out[n], n*DIM);
  //}

  // printf("Fence");
  gemmini_fence();

  // check equal
  //for (size_t n = 0; n < N; ++n)
    if (!is_equal(&InDense[0], &Out[0])) {
      printf("Matrix %u:\n", n);
      printMatrix(InDense);
      printf("Matrix %u output:\n", n);
      printMatrix(Out);
      printf("\n");

      exit(1);
   // }

  exit(0);
}

