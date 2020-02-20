// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  // printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);

  // printf("Initialize our input and output matrices in main memory\n");
  elem_t In[DIM][DIM];
  elem_t Out[DIM][DIM];
  elem_t Identity[DIM][DIM];
  elem_t Correct[DIM][DIM];

  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      In[i][j] = 10 + (i * DIM + j) % 64;
      Identity[i][j] = -(i == j);
      Correct[i][j] = -In[i][j];
    }

  // printf("Calculate the scratchpad addresses of all our matrices\n");
  // printf("  Note: The scratchpad is \"row-addressed\", where each address contains one matrix row\n");
  size_t In_sp_addr = 0;
  size_t Out_sp_addr = BANK_ROWS;
  size_t Identity_sp_addr = 2*BANK_ROWS;

  // printf("Move \"In\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_mvin(In, In_sp_addr);

  // printf("Move \"Identity\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_mvin(Identity, Identity_sp_addr);

  // printf("Multiply \"In\" matrix with \"Identity\" matrix with a bias of 0\n");
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 0);
  gemmini_preload(Identity_sp_addr, Out_sp_addr);
  gemmini_compute_preloaded(In_sp_addr, GARBAGE_ADDR);

  // printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  gemmini_mvout(Out, Out_sp_addr);

  // printf("Fence till Gemmini completes all memory operations\n");
  gemmini_fence();

  // printf("Check whether \"In\" and \"Out\" matrices are identical\n");
  if (!is_equal(Correct, Out)) {
    printf("Output matrix is not negative one multiplied by the input matrix!\n");
    printf("\"In\" matrix:\n");
    printMatrix(In);
    printf("\"Out\" matrix:\n");
    printMatrix(Out);
    printf("\n");

    printf("FAIL\n");
    exit(1);
  }

  printf("Output matrix is correct\nPASS\n");
  exit(0);
}

