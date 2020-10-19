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

#define KDIM 6

typedef union {
  float f;
  uint32_t bits;
} float_cast;

#ifdef ELEM_T_IS_LOWPREC_FLOAT
#define toelem( a, b ) \
{ \
    float_cast tmp = { (a) }; \
    (b) = (elem_t)ROUNDING_RIGHT_SHIFT_BITS(tmp.bits, (23 - (ELEM_T_SIG_BITS - 1))); \
}
#else
#define toelem( a, b )  (b) = (a)
#endif

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
  elem_t InA[KDIM][DIM];
  elem_t InB[KDIM][DIM];
  acc_t RefInA[KDIM][DIM];
  acc_t RefInB[KDIM][DIM];
  acc_t Bias[DIM][DIM];
  elem_t Out[DIM][DIM];
  acc_t OutGoldAcc[DIM][DIM];
  elem_t OutGold[DIM][DIM];

  scale_acc_t A_scale_factor = (rand() % 2) + 1;

  for (size_t i = 0; i < KDIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      RefInA[i][j] = rand() % 10;
      RefInB[i][j] = rand() % 10;
      toelem(RefInA[i][j], InA[i][j]);
      toelem(RefInB[i][j], InB[i][j]);
    }
  
  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      Bias[i][j] = rand() % 10;
    }

  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      OutGoldAcc[i][j] = Bias[i][j]; 
      for (size_t k = 0; k < KDIM; k++) {
        OutGoldAcc[i][j] += A_scale_factor*RefInA[k][i]*RefInB[k][j];
      }
      toelem(OutGoldAcc[i][j], OutGold[i][j]);
    }
  }

  printf("Calculate the scratchpad addresses of all our matrices\n");
  printf("  Note: The scratchpad is \"row-addressed\", where each address contains one matrix row\n");
  size_t InA_sp_addr = 0;
  size_t InB_sp_addr = 4*KDIM;
  const uint32_t Bias_sp_addr = 1 << (ADDR_LEN-1);
  const uint32_t Out_sp_addr = 3 << (ADDR_LEN-2);

  printf("Move \"Bias\" matrix from main memory into Gemmini's accumulators\n");
  gemmini_config_ld(DIM * sizeof(acc_t));
  gemmini_mvin(Bias, Bias_sp_addr);

  for (size_t K0 = 0; K0 < KDIM; K0+=DIM) {

    printf("Move \"InA\" matrix from main memory into Gemmini's scratchpad\n");
    gemmini_extended_config_ld(DIM * sizeof(elem_t), A_scale_factor);
    //gemmini_config_ld(DIM * sizeof(elem_t));
    gemmini_mvin(InA+K0, InA_sp_addr+K0);
  
    printf("Move \"InB\" matrix from main memory into Gemmini's scratchpad\n");
    gemmini_config_ld(DIM * sizeof(elem_t));
    gemmini_mvin(InB+K0, InB_sp_addr+K0);
  
    printf("Multiply \"InA\" transposed matrix with \"InB\" matrix\n");
    gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 0, 1, true, false)
    // gemmini_extended_config_ex(OUTPUT_STATIONARY, 0, 0, 0, 0, 1, false, false)
    
    gemmini_preload_zeros(K0 + DIM >= KDIM ? Out_sp_addr : GARBAGE_ADDR);
    if (K0 == 0) { // First iteration
        gemmini_extended_compute_preloaded(InA_sp_addr+K0, InB_sp_addr+K0, (K0 + DIM <= KDIM) ? DIM : KDIM - K0, DIM,
                                                                                   DIM, (K0 + DIM <= KDIM) ? DIM : KDIM - K0);
    } else { // All other iterations
        gemmini_extended_compute_accumulated(InA_sp_addr+K0, InB_sp_addr+K0, (K0 + DIM <= KDIM) ? DIM : KDIM - K0, DIM,
                                                                                 DIM, (K0 + DIM <= KDIM) ? DIM : KDIM - K0);
    }

  }

  printf("Move \"Out\" matrix from Gemmini's scratchpad into main memory\n");
  gemmini_mvout(Out, Out_sp_addr);

  printf("Fence till Gemmini completes all memory operations\n");
  gemmini_fence();

  printf("Check whether \"Out\" and \"Gold\" matrices are identical\n");

  if (!is_equal(Out, OutGold)) {
    printf("Ouput and Gold matrices are different!\n");
    // printf("\"InA\" matrix:\n");
    // printMatrix(InA);
    // printf("\"InB\" matrix:\n");
    // printMatrix(InB);
    // printf("\"Bias\" matrix:\n");
    // printMatrix(Bias);
    printf("\"Out\" matrix:\n");
    printMatrix(Out);
    printf("\"OutGoldAcc\" matrix:\n");
    printMatrixAcc(OutGoldAcc);
    printf("\"OutGold\" matrix:\n");
    printMatrix(OutGold);
    printf("\n");

    printf("\nShifted = %x\n", 0x43838000 >> (23 - 7));
    printf("Shifted and rounded = %x\n", ROUNDING_RIGHT_SHIFT_BITS(0x43838000, (23 - 7)));
    printf("Shifted and rounded 16 = %x\n", ROUNDING_RIGHT_SHIFT_BITS(0x43838000, 16));

    exit(1);
  }

  printf("Output and Gold matrices are identical, as expected\n");
  exit(0);
}

