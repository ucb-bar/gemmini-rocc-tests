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

void print_tile(elem_t* in, int tile_dim, int in_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*in_dim);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d.%d ", (int)(*(in +r*in_dim + c)), ((int)((*(in +r*in_dim + c))*100))%100);
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

void full_transposed_matmul(int block_size, int A_dim, int B_dim, int C_dim, elem_t* A, elem_t* B, elem_t* C, bool sub) {
  for (size_t r = 0; r < block_size; r++)
    for (size_t c = 0; c < block_size; c++) {
		elem_t temp = 0;
      for (size_t k = 0; k < block_size; k++){
			temp += (*(A+r*A_dim+k))*(*(B+c*B_dim+k));
		
		}
		if(!sub) *(C+C_dim*r+c) = temp;
		else *(C+C_dim*r+c) -= temp;	
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
      if (((int)(x[i][j]*50)) != ((int)(y[i][j]*50))){
			printf("i: %d, j: %d, value: %d.%d, %d.%d \n", i, j, (int)x[i][j], ((int)(x[i][j]*1000))%1000 , (int)y[i][j], ((int)(y[i][j]*1000))%1000);
      //   return 0;
		}
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

void lower_triangle_inverse(int block_size, elem_t* A, elem_t M[block_size][block_size]){
//	for(int i = 0; i < block_size; i++)
//		for(int j = 0; j < block_size; j++)
//			M[i][j] = 0;

	for(int i = 0; i < block_size; i++){
		M[i][i] = 1/(*(A+i*MAT_DIM+i));
		for(int j = 0; j < i; j++)
			M[j][i] = 0;
		for(int j = i+1; j < block_size; j++){
			elem_t sum = 0;
			M[j][i] = 0;
			for(int k = i; k < j; k++){
	//			if(j == i+1 && k != i) M[k][i] = 0;
				sum += (*(A+j*MAT_DIM+k))*M[k][i];///(*(A+j*MAT_DIM+j));
			}
			M[j][i] = M[j][i] - sum/(*(A+j*MAT_DIM+j));
		}
	}
}

//store transposed inverse for vector
void lower_triangle_inverse_transpose(int block_size, elem_t* A, elem_t M[block_size][block_size]){
	for(int i = 0; i < block_size; i++)
		for(int j = 0; j < block_size; j++)
			M[i][j] = 0;

	for(int i = 0; i < block_size; i++){
		M[i][i] = 1/(*(A+i*MAT_DIM+i));
		for(int j = i+1; j < block_size; j++){
			elem_t sum = 0;
			for(int k = i; k < j; k++)
				sum += (*(A+j*MAT_DIM+k))*M[i][k];///(*(A+j*MAT_DIM+j));
			M[i][j] = M[i][j] - sum/(*(A+j*MAT_DIM+j));
		}
	}
}
void full_right_chol(int block_size, elem_t* L){
	for(int k = 0; k < block_size; k++){
		//printf("%d %d \n", (int)(L[k][k]*100), (int)((float)(sqrt(L[k][k]))*100));
		*(L+k*MAT_DIM+k) = (float)(sqrt(*(L+k*MAT_DIM+k)));
		for(int i = 0; i < block_size; i++){
			if(i > k) *(L+i*MAT_DIM+k) = (float)(*(L+i*MAT_DIM+k) / *(L+k*MAT_DIM+k));
			else if(i < k) *(L+i*MAT_DIM+k) = 0;
		}
		for(int j = k+1; j < block_size; j++)
			for(int i = j; i < block_size; i++){
				//if(i==block_size-1 && j==block_size-1) printf("Lkk: %d, Lik:%d, Ljk: %d, mult: %d \n", (int)(L[i][j]*100), (int)(L[i][k]*100), (int)(L[j][k]*100), (int)(L[i][k]*L[j][k]*100));
				*(L+i*MAT_DIM+j) -= (*(L+i*MAT_DIM+k))*(*(L+j*MAT_DIM+k));
			}
				//printf("%d \n", (int)(L[k][k]));
	}
}

void copy_matrix(int block_size, int A_dim, int B_dim, elem_t* A, elem_t* B){
	for(int i = 0; i < block_size; i++)
		for(int j = 0; j < block_size; j++)
			*(B+i*B_dim+j) = *(A+i*A_dim+j);
}

void block_right_chol(int block_size, elem_t* L){
	for(int k = 0; k < num_block; k++){
		full_right_chol(block_size, L+block_size*(MAT_DIM*k+k));
		elem_t temp[block_size][block_size];
		lower_triangle_inverse(block_size, L+(k*MAT_DIM+k)*block_size, temp);
		for(int i = k+1; i < num_block; i++){	
			elem_t temp_mult[block_size][block_size];
			full_transposed_matmul(block_size, MAT_DIM, block_size, block_size, L+block_size*(i*MAT_DIM+k), (elem_t*) temp, (elem_t*) temp_mult, false);	
			copy_matrix(block_size, block_size, MAT_DIM, (elem_t*) temp_mult, L+block_size*(i*MAT_DIM+k));
		}
		for(int j = k+1; j < num_block; j++)
			for(int i = j; i < num_block; i++){
				full_transposed_matmul(block_size, MAT_DIM, MAT_DIM, MAT_DIM, L+block_size*(i*MAT_DIM+k), L+block_size*(j*MAT_DIM+k), L+block_size*(i*MAT_DIM+j), true);
			}	
	}
}

void block_left_chol(int block_size, elem_t* L){
	for(int k = 0; k < num_block; k++){
		for(int j = 0; j < k-1; j++)
			full_transposed_matmul(block_size, MAT_DIM, MAT_DIM, MAT_DIM, L+block_size*(k*MAT_DIM+k), L+block_size*(k*MAT_DIM+j), L+block_size*(k*MAT_DIM+j), true);
		full_right_chol(block_size, L+block_size*(MAT_DIM*k+k));
		elem_t temp_inv[block_size][block_size];
		lower_triangle_inverse(block_size, L+(k*MAT_DIM+k)*block_size, temp_inv);

		for(int i = k+1; i < num_block; i++){
			for(int j = 0; j < k-1; j++){
				full_transposed_matmul(block_size, MAT_DIM, MAT_DIM, MAT_DIM, L+block_size*(i*MAT_DIM+k), L+block_size*(i*MAT_DIM+j), L+block_size*(k*MAT_DIM+j), true);
			}
			elem_t temp[block_size][block_size];
			full_transposed_matmul(block_size, MAT_DIM, block_size, block_size, L+block_size*(i*MAT_DIM+k), (elem_t*) temp_inv, (elem_t*) temp, false);
		}
	}
}
	

void full_left_chol(int block_size, elem_t L[block_size][block_size]){
	for(int j=0; j < block_size; j++){
		for(int k=0; k < j; k++){
			L[j][j] -= L[j][k]*L[j][k];
			for(int i = j+1; i < block_size; i++)
				L[i][j] -= L[i][k]*L[j][k];
		}
		L[j][j] = (float)(sqrt(L[j][j]));
		for(int i = j+1; i < block_size; i++)
			L[i][j] = (float)(L[i][j]/L[j][j]);
		for(int i = 0; i < j; i++)
			L[i][j] = 0;
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
		 for(int j = 0; j < MAT_DIM; j++)
			 LR[j][k] = in_A[j][k];

	 printf("Starting naive right CPU chol\n");
    uint64_t cpu_start = read_cycles();
    full_right_chol(MAT_DIM, (elem_t *) LR);
    uint64_t cpu_end = read_cycles();
    printf("Cycles taken: %llu\n", cpu_end-cpu_start);

	 for(int j = 0; j < MAT_DIM; j++)
		 for(int jj = j; jj < MAT_DIM; jj++)
			 LL[jj][j] = in_A[jj][j];

	 printf("Starting naive left CPU chol\n");
	 cpu_start = read_cycles();
	 full_left_chol(MAT_DIM, LL);
	 cpu_end = read_cycles();
	 printf("Cycles taken: %llu\n", cpu_end-cpu_start);

/*
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
*/
	 for(int i = 0; i < MAT_DIM; i++)
		 for(int j = 0; j <= i; j++){
			 LR_block[i][j] = in_A[i][j];
			 LL_block[i][j] = in_A[i][j];
		 }

	 printf("Starting block right CPU chol\n");
    cpu_start = read_cycles();
    block_right_chol(block_dim, (elem_t *) LR_block);
    cpu_end = read_cycles();
    printf("Cycles taken: %llu\n", cpu_end-cpu_start);

	 printf("Starting block left CPU chol\n");
    cpu_start = read_cycles();
    block_right_chol(block_dim, (elem_t *) LL_block);
    cpu_end = read_cycles();
    printf("Cycles taken: %llu\n", cpu_end-cpu_start);


#endif


#if CHECK_RESULT == 1
    if (!full_is_equal(LR, gold_L)) {
      printf("C:\n");
//      full_printMatrix(LR);
      printf("Right Gold:\n");
//      full_printMatrix(gold_L);
      printf("\n");

      exit(1);
    }
   if (!full_is_equal(LL, gold_L)) {
      printf("C:\n");
//      full_printMatrix(LL);
      printf("Left Gold:\n");
//      full_printMatrix(gold_L);
      printf("\n");

      exit(1);
    }
	
    if (!full_is_equal(LR_block, gold_L)) {
      printf("C:\n");
//      full_printMatrix(LR_block);
      printf("Block Right Gold:\n");
//      full_printMatrix(gold_L);
      printf("\n");
		exit(1);

	 }
	 if (!full_is_equal(LL_block, gold_L)) {
      printf("C:\n");
//      full_printMatrix(LL_block);
      printf("Block Right Gold:\n");
//      full_printMatrix(gold_L);
      printf("\n");
      exit(1);
    }
#endif

  exit(0);
}

