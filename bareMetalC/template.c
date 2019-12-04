// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/systolic.h"

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

  printf("Flush Gemmini TLB of stale virtual addresses\n");
  matmul_flush(0);

  elem_t In[DIM][DIM];
  elem_t Out[DIM][DIM];

  printf("Move \"In\" matrix from main memory into Gemmini's scratchpad\n");
  matmul_mvin(In, DIM);
  printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  matmul_mvout(Out, DIM);

  printf("Fence till Gemmini completes all memory operations\n");
  matmul_fence();

  printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal(In, Out)) {
    printf("Input and output matrices are different!\n");
    printf("\"In\" matrix:\n");
    printMatrix(In);
    printf("\"Out\" matrix:\n");
    printMatrix(Out);
    printf("\n");

    exit(1);
  }

  printf("Input and output matrices are identical, as expected\n");
  exit(0);
}

