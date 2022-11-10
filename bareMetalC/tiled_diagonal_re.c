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
#define OP 3
#define FAST 1
#define CHECK_RESULT 1
#define NUM_ARRAY 2
#define ZONE 16

#define NO_BIAS 0
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#define MAT_DIM 16384 // (16x64x16)
#define MAT_DIM_S (MAT_DIM+64)

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], elem_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

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


void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
} 

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    static elem_t full_A[MAT_DIM][MAT_DIM_S] row_align(1);
    static elem_t full_B[MAT_DIM][MAT_DIM_S] row_align(1);
    static elem_t full_C[MAT_DIM][MAT_DIM_S] row_align(1);
    static ACC_T full_D[MAT_DIM][MAT_DIM_S] row_align_acc(1);

 //   static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
 //   static elem_t gold[MAT_DIM_I][MAT_DIM_J];
  printf("attempting rerocc_acquire\n");
  for(int i = 0; i < NUM_ARRAY; i++)
    while(!rerocc_acquire(i, 0xf)){}

  printf("rerocc acquired \n");

   
  for (int i = 0; i < NUM_ARRAY; i++) {
    rerocc_assign(OP, i);
    gemmini_flush(0);
  }
    printf("Starting gemmini matmul\n");
    unsigned long start = read_cycles();

    for(int z = 1; z < ZONE; z++){
       tiled_opcode_diagonal_auto(MAT_DIM, MAT_DIM, MAT_DIM,
            MAT_DIM_S, MAT_DIM_S, MAT_DIM_S, 
            true, true, true, true, // direct dram?
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            z,
            NUM_ARRAY, 0);
    }
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);

    // Release all the trackers
    for (int i = 0; i < NUM_ARRAY; i++) {
      rerocc_release(i);
    }

/*
    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();
#ifdef FAST
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        gold[i][j] = MAT_DIM_K + (NO_BIAS ? 0 : (RAND % 2));
      }
    }

#else
    full_matmul(full_A, full_B, full_D, gold);
#endif
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    //full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif
*/
#if CHECK_RESULT == 1
    if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      full_printMatrix(full_C);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

