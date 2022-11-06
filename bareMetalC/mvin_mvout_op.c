// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/rerocc.h" 
#include "include/gemmini_testutils.h"

#define N 4*3
#define OP 2

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

  int r = rerocc_ntrackers();
  printf("number of trackers: %d\n", r);
  printf("attempting rerocc_acquire\n");
  for(int i = 0; i < 3; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");

  printf("Flush\n");

  rerocc_assign(1, 0);
  rerocc_assign(2, 1);
  rerocc_assign(3, 2);

  gemmini_opcode_flush(1, 0);
  gemmini_opcode_config_ld(1, DIM*sizeof(elem_t));
  gemmini_opcode_config_st(1, DIM*sizeof(elem_t));

  gemmini_opcode_flush(2, 0);
  gemmini_opcode_config_ld(2, DIM*sizeof(elem_t));
  gemmini_opcode_config_st(2, DIM*sizeof(elem_t));


  gemmini_opcode_flush(3, 0);
  gemmini_opcode_config_ld(3, DIM*sizeof(elem_t));
  gemmini_opcode_config_st(3, DIM*sizeof(elem_t));


  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;



  for (size_t n = 0; n < N; n+=3) {
      printf("Mvin %d i %d\n", n, 0);
      gemmini_opcode_mvin(1, In[n], n*DIM);
      printf("Mvin %d i %d\n", n, 1);
      gemmini_opcode_mvin(2, In[n+1], n*DIM);
      printf("Mvin %d i %d\n", n, 2);
      gemmini_opcode_mvin(3, In[n+2], n*DIM);
      printf("Mvout %d i %d\n", n, 0);
      gemmini_opcode_mvout(1, Out[n], n*DIM);
      printf("Mvout %d i %d\n", n, 1);
      gemmini_opcode_mvout(2, Out[n+1], n*DIM);
      printf("Mvout %d i %d\n", n, 2);
      gemmini_opcode_mvout(3, Out[n+2], n*DIM);
      gemmini_fence();
  }
  // Release all the trackers
  for (int i = 0; i < 3; i++) {
    rerocc_release(i);
  }

  printf("rerocc released\n");
  for (size_t n = 0; n < N; ++n)
    if (!is_equal(In[n], Out[n])) {
      //printf("Matrix %u:\n", n);
      //printMatrix(In[n]);
      printf("Matrix %u output:\n", n);
      printMatrix(Out[n]);
      printf("\n");

      //exit(1);
    }

  exit(0);
}

