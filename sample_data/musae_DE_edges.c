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
#include "musae_DE_edges.h"

#define CHECK_RESULT 1

#define NO_BIAS false
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#define MAT_DIM_I dim_I
#define MAT_DIM_K 256
#define MAT_DIM_J dim_J

//dot product
void full_matdot(elem_t A[MAT_DIM_I][MAT_DIM_J], elem_t B[MAT_DIM_I][MAT_DIM_J], elem_t C[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) 
	C[r][c] = A[r][c]*B[r][c];
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
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

void half_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I/2; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matshift(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Bitshift and round element
      full_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
#else
      out[r][c] = shifted; // TODO should we also saturate when using floats?
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

  printf("Flush Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(0);

  printf("Initialize our input and output matrices in main memory\n");
  elem_t A[MAT_DIM_I][MAT_DIM_K];
  elem_t B[MAT_DIM_K][MAT_DIM_J];
  elem_t C[MAT_DIM_I][MAT_DIM_J];
//  full_t gold_raw[MAT_DIM_I][MAT_DIM_J];
  elem_t gold_dense[MAT_DIM_I][MAT_DIM_J];
//  elem_t S[MAT_DIM_I][MAT_DIM_J];
//  elem_t gold[MAT_DIM_I][MAT_DIM_J];


	for(int j = 0; j < MAT_DIM_I; j++){
		for(int i = 0; i < MAT_DIM_K; i++){
			A[j][i] = rand() % 3 - 1;
		}
	}

	for(int j = 0; j < MAT_DIM_K; j++){
		for(int i = 0; i < MAT_DIM_J; i++){
			B[j][i] = rand() % 3 - 1;
		}
	}


  acc_t full_D[MAT_DIM_I][MAT_DIM_J];

  for (size_t i = 0; i < MAT_DIM_K; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++){
      full_D[i][j] = 0;
    }

	
//normal dense matmul
    printf("Starting dense matmul \n");
    unsigned long dense_start = read_cycles();
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)gold_dense,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
            NO_ACTIVATION, 0, 0, false,
            WS);
    unsigned long dense_end = read_cycles();
    printf("cycles taken: %u\n", dense_end - dense_start);

//  full_matmul(A, B, full_D, gold_raw); //for result check
//  full_matshift(gold_raw, gold_dense, 0);
//  full_matdot(gold_dense, S, gold);


    //automatic tiling
    printf("Starting SDDMM \n");
    unsigned long sddmm_start = read_cycles();
    tiled_matmul_auto_sddmm(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)C, (int*) S_indptr, (int*) S_index, (int*) S_indptr_j, (int*) S_index_j,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
            NO_ACTIVATION, 0, 0, false,
            WS);
    unsigned long sddmm_end = read_cycles();
    printf("cycles taken: %u\n", sddmm_end - sddmm_start);

/*
  if (!full_is_equal(gold, C)) {
    printf("expected and real matrices are different!\n");
    printf("\"dense\" matrix:\n");
    full_printMatrix(gold_dense);
    printf("\"Sampling\" matrix:\n"); 
    full_printMatrix(S); 
    printf("\"Out\" matrix:\n");
    full_printMatrix(C);
    printf("\"gold\" matrix:\n");
    full_printMatrix(gold);
    printf("\n");
    exit(1);
  }
*/
//  printf("expected and real matrices are identical, as expected\n");
  exit(0);
}

