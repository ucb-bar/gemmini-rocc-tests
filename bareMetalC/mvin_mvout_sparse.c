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

#define N 1
#define SPARSITY 10

#if (N*DIM) > (BANK_NUM*BANK_ROWS)
#error not enough scratchpad space
#endif

void printSparseMat(elem_t data[DIM*DIM], ind_t coo[DIM*DIM][2]) {
  for (size_t i = 0; i < DIM*DIM; ++i) {
#ifndef ELEM_T_IS_FLOAT
      if (data[i])
          printf("[%d] %d (%d, %d), ", i, data[i], coo[i][0], coo[i][1]);
#else
      if (data[i])
          printf("[%d] %x (%d, %d), ", i, elem_t_to_elem_t_bits(data[i]), coo[i][0], coo[i][1]);
#endif
  }
  printf("\n");
}

void printFMatrix(float m[DIM][DIM]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j)
      printf("%f ", m[i][j]);
    printf("\n");
  }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  printf("Flush\n");
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  // create
  static elem_t InDense[N][DIM][DIM] row_align(1);
  static elem_t OutDense[N][DIM][DIM] row_align(1);

  // the current mvin requires that this is 1 element larger than the number of nonzero values to avoid segfaults
  static elem_t InSpData[N][DIM*DIM]; row_align(1);  // it can be smaller, but I'm maxing it for now
  static ind_t InSpCoo[N][DIM*DIM][2]; row_align(1); 
  
  printf("pre-populate\n");
  unsigned int sparse_index = 0;
  for (size_t n = 0; n < N; ++n) {
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        if ((rand() % 100) < SPARSITY) { 
          //int value = (i*DIM + j + n); // / 2; there is some sort of clamping going on here
          float value = (float) (rand() % 128); 
          InDense[n][i][j]          = value;
          InSpData[n][sparse_index] = value;
          InSpCoo[n][sparse_index][0] = i;
          InSpCoo[n][sparse_index][1] = j;
          sparse_index += 1;
        }
    printf("Input Matrix[%d]\n",n);
    printMatrix(InDense[n]);
    printSparseMat(InSpData[n], InSpCoo[n]);
  }

  printf("Addr SpData: %p %p %p\n", InSpData, &InSpData[0], &InSpData[0][0]);
  printf("Addr SpCoo : %p %p %p\n", InSpCoo, &InSpCoo[0], &InSpCoo[0][1]);
  printf("InSpData[0] %d\n", ((elem_t*) InSpData)[0]);
  printf("InSpData[0] %d\n", InSpData[0][0]);
  printf("InSpCoo[0]  (%d,%d)\n", InSpCoo[0][0][0], InSpCoo[0][0][1]);
  printf("InDnData[0] %d\n", InDense[0][0][0]);

  printf("Fence");
  gemmini_fence();

  // Apply mvin mvout
  for (size_t n = 0; n < N; ++n) {
    printf("Mvin %d\n", n);
    // the current spike implementation requires that the addresses of data and coo passed in 
    // point to the first nonzero value after or including (start_row, start_col) (in this case 0,0)
    gemmini_extended_mvin_sparse_coo(&InSpData[n], &InSpCoo[n], DIM*DIM, n*DIM, 0, DIM, 0, DIM); 
    printf("Mvout %d\n", n);
    gemmini_mvout(&OutDense[n], n*DIM);
  }

  printf("Fence\n");
  gemmini_fence();

  // check equal
  for (size_t n = 0; n < N; ++n) {
    if (!is_equal(&InDense[0], &OutDense[0])) {
      printf("Matrix %u:\n", n);
      printMatrix(InDense);
      printf("Matrix %u output:\n", n);
      printMatrix(OutDense);
      printf("\n");
      exit(1);
    } else {
      printf("Comp[%d] PASS\n", n);
    }
  }

  exit(0);
}

