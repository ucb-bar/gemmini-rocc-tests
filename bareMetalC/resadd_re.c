// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
//#include "include/rerocc.h" 
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1
#define OP 3

#ifndef BAREMETAL
#define MAT_DIM_I 512
#define MAT_DIM_J 512
#else
#define MAT_DIM_I 64
#define MAT_DIM_J 64
#endif

#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true
#define NUM_ARRAY 4
#define NUM_ARRAY1 1

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j]){
        printf("i: %d, j: %d, x: %d, y: %d\n", i, j, x[i][j], y[i][j]); 
        //return 0;
      }
  return 1;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
  int r = rerocc_ntrackers();
  printf("number of trackers: %d\n", r);
    static elem_t A[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t B[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
    // printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 2);//(rand() % 64) - 32;
        B[i][j] = (rand() % 3); //(rand() % 8) - 4;
        gold[i][j] = A[i][j]+B[i][j];//1+(i%10);
      }
    }
/*
#if CHECK_RESULT == 1
    printf("Starting slow CPU resadd\n");
    unsigned long cpu_start = read_cycles();
    resadd_cpu(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, (elem_t*)A, (elem_t*)B,
            (elem_t*)gold, USE_RELU);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
#endif
*/
#endif
    unsigned long start = 0;
    unsigned long end = 0;
    
  printf("attempting rerocc_acquire\n");
  for(int i = 0; i < NUM_ARRAY; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");


  for (int i = 0; i < NUM_ARRAY; i++) {
    rerocc_assign(OP, i);
    gemmini_flush(0);
  } 
    printf("Starting gemmini resadd\n");
    start = read_cycles();
    tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
            (elem_t*)A, (elem_t*)B,
            (elem_t*)C, USE_RELU, WS, NUM_ARRAY, 0);
    end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    // Release all the trackers
    for (int i = 0; i < NUM_ARRAY; i++) {
      rerocc_release(i);
    }

    printf("rerocc released\n");

#if CHECK_RESULT == 0
    static elem_t A1[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t B1[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static elem_t C1[MAT_DIM_I][MAT_DIM_J] row_align(1);
   printf("attempting rerocc_acquire\n");
  for(int i = 0; i < 1; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");

   
  for (int i = 0; i < NUM_ARRAY1; i++) {
    rerocc_assign(OP, i);
    gemmini_flush(0);
  } 
    printf("Starting single gemmini resadd\n");
    start = read_cycles();
    tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
            (elem_t*)A1, (elem_t*)B1,
            (elem_t*)C1, USE_RELU, WS, NUM_ARRAY1, 0);
    end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    // Release all the trackers
    for (int i = 0; i < NUM_ARRAY1; i++) {
      rerocc_release(i);
    }

    printf("rerocc released\n");
#endif
#if CHECK_RESULT == 1
    if (!full_is_equal(C, gold)) {
     // printf("C:\n");
     // full_printMatrix(C);
     // printf("Gold:\n");
     // full_printMatrix(gold);
     // printf("A:\n");
     // full_printMatrix(A);
     // printf("B:\n");
     // full_printMatrix(B);
     // printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

