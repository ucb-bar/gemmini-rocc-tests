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

#define CHECK_RESULT 1

#define NO_BIAS false
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#define UNIT 32
#define MAT_DIM_I 48
#define MAT_DIM_K 32
#define MAT_DIM_J 48
#define tile_I UNIT/DIM
#define tile_J UNIT/DIM
#define tile_K UNIT/DIM

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
  elem_t gold_raw[MAT_DIM_I][MAT_DIM_J];
  elem_t gold_dense[MAT_DIM_I][MAT_DIM_J];
  elem_t S[MAT_DIM_I][MAT_DIM_J];
  elem_t gold[MAT_DIM_I][MAT_DIM_J];


	for(int j = 0; j < MAT_DIM_I; j++){
		for(int i = 0; i < MAT_DIM_K; i++){
			A[j][i] = 1;//rand() % 2;
		}
	}

	for(int j = 0; j < MAT_DIM_K; j++){
		for(int i = 0; i < MAT_DIM_J; i++){
			B[j][i] = 1;//rand() % 2;
		}
	}

	//sparse matrix
	for(int j = 0; j < MAT_DIM_I; j++){
		for(int i = 0; i < MAT_DIM_J; i++){
			if(rand()%100 == 1){
				S[j][i] = 1;
			}
			else S[j][i] = 0;
		}
	}

  acc_t full_D[MAT_DIM_I][MAT_DIM_J];

  for (size_t i = 0; i < MAT_DIM_K; i++)
    for (size_t j = 0; j < MAT_DIM_J; j++){
      full_D[i][j] = 0;
    }

	int S_indptr[MAT_DIM_I+1];
	int S_index[MAT_DIM_J*MAT_DIM_I];//length?
	int S_indptr_j[MAT_DIM_J+1];
	int S_index_j[MAT_DIM_J*MAT_DIM_I];//length?
	S_indptr[0] = 0;
	S_indptr_j[0] = 0;

	for(int j = 0; j < MAT_DIM_I; j++){
		for(int i = 0; i < MAT_DIM_J; i++){
			if(S[j][i] != 0) S_indptr[j+1]++;
		}
		if(j < MAT_DIM_I-1) S_indptr[j+2] = S_indptr[j+1];
	}
 	
	for(int j = 0; j < MAT_DIM_J; j++){
		for(int i = 0; i < MAT_DIM_I; i++){
			if(S[i][j] != 0) S_indptr_j[j+1]++;
		}
		if(j < MAT_DIM_J-1) S_indptr_j[j+2] = S_indptr_j[j+1];
	}
 	

	int row = 0;
	int num = 0;
//	printf("index i: \n");
	for(int j = 0; j < MAT_DIM_I; j++){
		for(int i = 0; i < MAT_DIM_J; i++){
			if(S[j][i] != 0){
				S_index[num] = i; //column index
				//Out[row+DIM/2][num] = In[j][i]; //store data value
//				printf("%d, ", S_index[num]);
				num ++;
			}
		}
	}
//	printf("\n");
	row = 0;
	num = 0;
//	printf("index j: \n");
        for(int j = 0; j < MAT_DIM_J; j++){
		for(int i = 0; i < MAT_DIM_I; i++){
 
			if(S[i][j] != 0){
				S_index_j[num] = i; //row index
				//Out[row+DIM/2][num] = In[j][i]; //store data value
//				printf("%d, ", S_index_j[num]);
				num ++;
			}
		}
	}
//	printf("\n");

//normal dense matmul
    printf("normal dense matmul \n");
    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)gold_raw,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
            NO_ACTIVATION, 0, 0, false,
            WS);

//  full_matmul(A, B, full_D, gold_raw); //for result check
//  full_matshift(gold_raw, gold_dense, 0);
//  full_matdot(gold_dense, S, gold);

	for(int i = 0; i < MAT_DIM_I; i++){
		int start = *(S_indptr + i);
		int end = *(S_indptr + i + 1);
		for (int j = start; j < end; j++){
			int index = *(S_index + j);
			gold[i][index] = gold_raw[i][index];
		}
	}




/*
    //automatic tiling
    tiled_matmul_auto_sddmm(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)C, (int*) S_indptr, (int*) S_index, (int*) S_indptr_j, (int*) S_index_j,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
            NO_ACTIVATION, 0, 0, false,
            WS);

*/
   printf("sddmm matmul \n");
    tiled_matmul_sddmm(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)C, (int*) S_indptr, (int*) S_index, (int*) S_indptr_j, (int*) S_index_j, 
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_ONE, MVIN_SCALE_ONE, MVIN_SCALE_ONE,
            NO_ACTIVATION, 0, 0, false,
	    tile_I, tile_J, tile_K,
            WS);

  if (!full_is_equal(gold, C)) {
    printf("expected and real matrices are different!\n");
//    printf("\"dense\" matrix:\n");
//    full_printMatrix(gold_dense);
//    printf("\"Sampling\" matrix:\n"); 
//    full_printMatrix(S); 
    printf("\"Out\" matrix:\n");
    full_printMatrix(C);
    printf("\"gold\" matrix:\n");
    full_printMatrix(gold);
    printf("\n");
    exit(1);
//  }
 }

/*

//dense
  if (!full_is_equal(gold_shift, Out)) {
    printf("expected and real matrices are different!\n");
    printf("\"In_comp\" matrix:\n");
    half_printMatrix(In);
    printf("\"B_comp\" matrix:\n"); 
    half_printMatrix(B); 
    printf("\"Out\" matrix:\n");
    half_printMatrix(Out);
    printf("\"gold\" matrix:\n");
    half_printMatrix(gold_shift);
    printf("\n");
    exit(1);
  }  
*/

  printf("expected and real matrices are identical, as expected\n");
  exit(0);
}

