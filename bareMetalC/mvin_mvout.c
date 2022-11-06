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

#define N 4*4
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
  for(int i = 0; i < 4; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");

  for (int i = 0; i < 4; i++) {
    rerocc_assign(OP, i);
    gemmini_flush(0);
    gemmini_config_ld(DIM * sizeof(elem_t));
    gemmini_config_st(DIM * sizeof(elem_t));
  }

  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);

  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j)
        In[n][i][j] = i*DIM + j + n;



  for (size_t n = 0; n < N; n+=4) {
    for(size_t i = 0; i < 4; i++){
      rerocc_assign(OP, i);
      int ni = n+i;
      printf("Mvin %d i %d\n", n, i);
      gemmini_mvin(In[ni], n*DIM);
      printf("Mvout %d i %d\n", n, i);
      gemmini_mvout(Out[ni], n*DIM);
      printf("mvout done\n");
      gemmini_fence();
    }
  }
  // Release all the trackers
  for (int i = 0; i < 4; i++) {
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

