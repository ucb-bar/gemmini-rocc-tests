
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"


int main() {

  printf("running a very basic matrix add -- from matrix_add.c\n");

  elem_t A[DIM][DIM]; // typedef int8_t elem_t from gemmini_param.h, DIM=16
  elem_t B[DIM][DIM];
  elem_t C[DIM][DIM];
  elem_t gold[DIM][DIM];


  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++) {
      A[i][j] = (rand() % 16) - 8;
      B[i][j] = (rand() % 16) - 8;
    }


  int ascale = 1;
  int bscale = 1;

  // golden model
  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      acc_t sum = MVIN_SCALE(A[i][j], ascale) + MVIN_SCALE(B[i][j], bscale); // typedef int32_t acc_t from gemmini_param.h
      gold[i][j] = sum > elem_t_max ? elem_t_max :
        (sum < elem_t_min ? elem_t_min : sum); // where is elem_t_max/min defined?
    }
  }

  // calling gemmini ISA
  
  printf("Starting gemmini mat add\n");
  unsigned long start = read_cycles();
  uint32_t A_acc_addr = 1 << (ADDR_LEN - 1); // ADDR_LEN=32 from gemmini_param.h
  uint32_t B_acc_addr = (1 << (ADDR_LEN - 1)) | (1 << (ADDR_LEN - 2)); // why?
  uint32_t C_acc_addr = 1 << (ADDR_LEN - 1);

  gemmini_extended2_config_ld(DIM * sizeof(elem_t), ascale, true);  // (stride, scale, shrunk)
  gemmini_mvin(A, A_acc_addr); // (dram_addr, spad_addr)

  gemmini_extended2_config_ld(DIM * sizeof(elem_t), bscale, true);  // (stride, scale, shrunk)
  gemmini_mvin(B, B_acc_addr); // (dram_addr, spad_addr)

  
  gemmini_config_ex(0, NO_ACTIVATION, 0); // (dataflow, sys_act, sys_shift)

  // how is the accumulator operating to do the matrix sum? C = A + B or spad[1<<31] <= spad[1<<31] + spad[1<<31+1<<30]


  gemmini_config_st(DIM * sizeof(elem_t)); // (stride) or (stride, NO_ACTIVATION, ACC_SCALE_IDENTITY) 
  gemmini_mvout(C, C_acc_addr); // (dram_addr, spad_addr)

  gemmini_fence();
  
  unsigned long end = read_cycles();
  printf("Cycles taken: %u\n", end-start);

  if (!is_equal(C, gold)) {
    printf("you're wrong\n");
    exit(1);
  }
  else {
    printf("you're on the right track!\n");
    exit(0);
  }


  
}
