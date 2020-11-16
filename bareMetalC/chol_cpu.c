// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "chol_data.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif


void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%f ", *(in +r*MAT_DIM + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM][MAT_DIM], elem_t B[MAT_DIM][MAT_DIM], ACC_T D[MAT_DIM][MAT_DIM], full_t C_full[MAT_DIM][MAT_DIM]) {
  for (size_t r = 0; r < MAT_DIM; r++)
    for (size_t c = 0; c < MAT_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*100))%100);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM][MAT_DIM], elem_t y[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i)
    for (size_t j = 0; j < MAT_DIM; ++j)
      if (((int)(x[i][j]*100)) != ((int)(y[i][j]*100)))
        return 0;
  return 1;
}

void full_matshift(full_t full[MAT_DIM][MAT_DIM], elem_t out[MAT_DIM][MAT_DIM], int shift) {
  for (size_t r = 0; r < MAT_DIM; r++)                             
    for (size_t c = 0; c < MAT_DIM; c++) {
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

void full_right_chol(int block_dim, elem_t* L){
	for(int k = 0; k < block_dim; k++){
		//printf("%d %d \n", (int)(L[k][k]*100), (int)((float)(sqrt(L[k][k]))*100));
		*(L+k*block_dim+k) = (float)(sqrt(*(L+k*block_dim+k)));
		for(int i = k+1; i < block_dim; i++)
			*(L+i*block_dim+k) = (float)(*(L+i*block_dim+k) / *(L+k*block_dim+k));
		for(int j = k+1; j < block_dim; j++)
			for(int i = j; i < block_dim; i++){
				//if(i==block_dim-1 && j==block_dim-1) printf("Lkk: %d, Lik:%d, Ljk: %d, mult: %d \n", (int)(L[i][j]*100), (int)(L[i][k]*100), (int)(L[j][k]*100), (int)(L[i][k]*L[j][k]*100));
				*(L+i*block_dim+j) -= (*(L+i*block_dim+k))*(*(L+j*block_dim+k));
			}
				//printf("%d \n", (int)(L[k][k]));
	}
}

void block_right_chol(int block_dim, elem_t* L){
	int num_block = MAT_DIM/block_dim;
	for(int k = 0; k < num_block; k++){
		full_right_chol(block_dim, L+k*MAT_DIM+k);
		for(int i = k+1; i < num_block; i++){
			lower_triangle_inverse(block_dim, L+k*MAT_DIM+k);
			
		}
	}
}

void full_left_chol(int block_dim, elem_t L[block_dim][block_dim]){
	for(int j=0; j < block_dim; j++){
		for(int k=0; k < j; k++){
			L[j][j] -= L[j][k]*L[j][k];
			for(int i = j+1; i < block_dim; i++)
				L[i][j] -= L[i][k]*L[j][k];
		}
		L[j][j] = (float)(sqrt(L[j][j]));
		for(int i = j+1; i < block_dim; i++)
			L[i][j] = (float)(L[i][j]/L[j][j]);
	}
}
/*
void lower_triangle_inverse(int block_dim, elem_t A[block_dim][block_dim]){
	elem_t M[block_dim][block_dim];
	for(int i = 0; i < block_dim; i++){
		M[i][i] = 1/A[i][i];
		for(int j = 0; j < i; j++){
			for(int k = j; k < i; k++)
				M[i][j] += A[i][k]*M[k][j];
			M[i][j] = -M[i][j]/A[i][i];
		}
	}
	for(int i = 0; i < block_dim; i++)
		for(int j = 0; j < block_dim; j++)
			A[i][j] = M[i][j];
}
*/
void lower_triangle_inverse(int block_dim, elem_t* A){
	elem_t M[block_dim][block_dim];
	for(int i = 0; i < block_dim; i++){
		M[i][i] = 1/(*(A+i*MAT_DIM+i));
		for(int j = 0; j < i; j++){
			for(int k = j; k < i; k++)
				M[i][j] += (*(A+i*MAT_DIM+k))*M[k][j];
			M[i][j] = -M[i][j]/(*(A+i*MAT_DIM+i));
		}
	}
	for(int i = 0; i < block_dim; i++)
		for(int j = 0; j < block_dim; j++)
			*(A+i*MAT_DIM+j) = M[i][j];
}
int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    static elem_t LR[MAT_DIM][MAT_DIM] row_align(1) = {0};
	 static elem_t LL[MAT_DIM][MAT_DIM] row_align(1) = {0};

    static elem_t LR_block[MAT_DIM][MAT_DIM] row_align(1) = {0};
	 static elem_t LL_block[MAT_DIM][MAT_DIM] row_align(1) = {0};

#if CHECK_RESULT == 1

	 for(int k = 0; k < MAT_DIM; k++)
		 for(int j = k; j < MAT_DIM; j++)
			 LR[j][k] = in_A[j][k];

	 printf("Starting naive right CPU chol\n");
    unsigned long cpu_start = read_cycles();
    full_right_chol(MAT_DIM, (elem_t *) LR);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);

	 for(int j = 0; j < MAT_DIM; j++)
		 for(int jj = j; jj < MAT_DIM; jj++)
			 LL[jj][j] = in_A[jj][j];

	 printf("Starting naive left CPU chol\n");
	 cpu_start = read_cycles();
	 full_left_chol(MAT_DIM, LL);
	 cpu_end = read_cycles();
	 printf("Cycles taken: %u\n", cpu_end-cpu_start);

	 int num_block = MAT_DIM/DIM;
	 int block_size = MAT_DIM/num_block;
	 for(int i = 0; i < num_block; i++)
		 for(int j = 0; j < num_block; j++)
			 for(int ii = 0; ii < block_size; ii++){
				 for(int jj = 0; jj < block_size; jj++){
					 if(j > i){
						 LR_block[i*block_size+ii][j*block_size+jj] = 0;
						 LL_block[i*block_size+ii][j*block_size+jj] = 0;
					 }
					 else{
						 LR_block[i*block_size+ii][j*block_size+jj] = in_A[i*block_size+ii][j*block_size+jj];
					    LL_block[i*block_size+ii][j*block_size+jj] = in_A[i*block_size+ii][j*block_size+jj];
					 }
				 }
			 }


#endif


#if CHECK_RESULT == 1
    if (!full_is_equal(LR, gold_L)) {
      printf("C:\n");
      full_printMatrix(LR);
      printf("Right Gold:\n");
      full_printMatrix(gold_L);
      printf("\n");

      exit(1);
    }
   if (!full_is_equal(LL, gold_L)) {
      printf("C:\n");
      full_printMatrix(LL);
      printf("Left Gold:\n");
      full_printMatrix(gold_L);
      printf("\n");

      exit(1);
    }
	
#endif

	lower_triangle_inverse(MAT_DIM, (elem_t*) LL);
#if CHECK_RESULT == 1
    if (!full_is_equal(LL, gold_invL)) {
      printf("inverse:\n");
      full_printMatrix(LL);
      printf("inverse gold:\n");
      full_printMatrix(gold_invL);
      printf("\n");

      exit(1);
    }
	
#endif


  exit(0);
}

