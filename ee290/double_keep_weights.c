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

#define N 2 // Number of matrix multiplications

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  // Flush Gemmini TLB of stale virtual addresses
  gemmini_flush(0);

  // Initialize our input and output matrices in main memory
  elem_t A[N][DIM][DIM];
  elem_t B[DIM][DIM]; // There is only one weight matrix this time, which we keep constant
  elem_t C[N][DIM][DIM];

  // Initialize A and B
  for (size_t n = 0; n < N; n++)
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++) {
        A[n][i][j] = (n+1)*10 + 2*(i*DIM + j);
      }

  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      B[i][j] = 10 + 2*(i*DIM + j) + 1;
    }

  // Calculate both C matrices
  for (size_t n = 0; n < N; n++)
    for (size_t i = 0; i < DIM; i++)
      for (size_t j = 0; j < DIM; j++) {
        int result = 0;

        for (size_t k = 0; k < DIM; k++) {
          result += A[n][i][k] * B[k][j];
        }

        // Gemmini will saturate results, instead of simply overflowing
        result = result < elem_t_max ? result : elem_t_max;
        result = result > elem_t_min ? result : elem_t_min;

        C[n][i][j] = result;
      }

  // Move in A and B matrices from main memory to Gemmini's scratchpad
  const size_t A_sp_addr = 0;
  const size_t B_sp_addr = BANK_ROWS;
  const size_t C_sp_addr = 2*BANK_ROWS;

  for (size_t n = 0; n < N; n++) {
    gemmini_mvin(A[n], A_sp_addr + n*DIM);
  }
  gemmini_mvin(B, B_sp_addr);

  // Multiply A matrices with B matrices in Gemmini;
  gemmini_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 0);

  // Calculate A[0] * B = C[0]
  gemmini_preload(B_sp_addr, C_sp_addr);
  gemmini_compute_preloaded(A_sp_addr, GARBAGE_ADDR);

  // Calculate A[1] * B = C[1]
  gemmini_preload(GARBAGE_ADDR, C_sp_addr + DIM);
  gemmini_compute_accumulated(A_sp_addr + DIM, GARBAGE_ADDR);

  // Move C matrices from Gemmini's scratchpad into main memory
  elem_t Out[N][DIM][DIM];
  for (size_t n; n < N; n++) {
    gemmini_mvout(Out[n], C_sp_addr + n*DIM);
  }

  // Fence till Gemmini completes all memory operations
  gemmini_fence();

  // Check whether Gemmini calculated C matrices correctly
  for (size_t n = 0; n < N; n++) {
    if (!is_equal(Out[n], C[n])) {
      printf("Output matrix %u is incorrect!\n", n);
      printf("What Gemmini calculated:\n");
      printMatrix(Out[n]);
      printf("The correct result:\n");
      printMatrix(C[n]);
      printf("\n");

      printf("FAIL\n");
      exit(1);
    }
  }

  printf("PASS\n");
  exit(0);
}

