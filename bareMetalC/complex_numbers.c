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

#define NEGATIVE_I (0x0000FFFF)
#define ONE (0x00010000)

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);

  printf("Initialize our input and output matrices in main memory\n");
  elem_t In[DIM][DIM];
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      In[i][j] = i == j ? NEGATIVE_I : 0;

  elem_t Out[DIM][DIM];

  printf("Calculate the scratchpad addresses of all our matrices\n");
  printf("  Note: The scratchpad is \"row-addressed\", where each address contains one matrix row\n");
  size_t In1_sp_addr = 0;
  size_t In2_sp_addr = DIM;
  size_t Out_sp_addr = 2*DIM;

  printf("Move \"In\" matrix from main memory into Gemmini's scratchpad\n");
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mvin(In, In1_sp_addr);
  gemmini_mvin(In, In2_sp_addr);

  printf("Multiply \"In\" matrix with \"In\" matrix with a bias of 0\n");
  gemmini_config_ex(OUTPUT_STATIONARY, 0, 0, 0);
  gemmini_preload_zeros(Out_sp_addr);
  gemmini_compute_preloaded(In1_sp_addr, In2_sp_addr);

  printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  gemmini_config_st(DIM * sizeof(elem_t));
  gemmini_mvout(Out, Out_sp_addr);

  printf("Fence till Gemmini completes all memory operations\n");
  gemmini_fence();

  printf("Input matrix was:\n");
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      int16_t real = In[i][j] >> 16;
      int16_t imag = In[i][j] & 0xFFFF;

      char sign = imag < 0 ? '-' : '+';
      int16_t abs_imag = image < 0 ? -imag : imag;

      printf("(%d %c %di) ", real, sign, abs_imag);
    }
    printf("\n");
  }

  printf("Output matrix (In * In) is:\n");
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      int16_t real = Out[i][j] >> 16;
      int16_t imag = Out[i][j] & 0xFFFF;

      char sign = imag < 0 ? '-' : '+';
      int16_t abs_imag = image < 0 ? -imag : imag;

      printf("(%d %c %di) ", real, sign, abs_imag);
    }
    printf("\n");
  }

  exit(0);
}

