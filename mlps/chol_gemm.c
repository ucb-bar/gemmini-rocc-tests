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
void full_matmul(elem_t A[MAT_DIM][MAT_DIM], elem_t B[MAT_DIM][MAT_DIM], ACC_T D[MAT_DIM][MAT_DIM], full_t C_full[MAT_DIM][MAT_DIM]) {
  for (size_t r = 0; r < MAT_DIM; r++)
    for (size_t c = 0; c < MAT_DIM; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_transposed_matmul(int I_block, int J_block, int K_block, int block_dim, int A_stride, int B_stride, int C_stride, elem_t* A, elem_t* B, elem_t* C, bool sub, int num_array) {
	/*
	tiled_opcode_matmul_auto_multi(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A_stride, B_stride, C_stride, C_stride,
            false, false, false, false,
			A, B, sub ? C : NULL, C,
			MVIN_SCALE_IDENTITY, sub ? (-1) :MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            false, true,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, false, true, 
			WS,
            num_array, 0);	
	*/
	    tiled_matmul_auto(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A, B, sub ? C : NULL, C,
			A_stride, B_stride, C_stride, C_stride,
            false, false, false, false,
			MVIN_SCALE_IDENTITY, sub ? (-1) :MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, true, false, 0,
			0, WS);	

}

// Lx = b (solve x)
// for partial matrix, update is whether to update the below part of b (togo_dimension)
void cpu_gaussian_elimination(elem_t* L, elem_t* x, elem_t* b, int L_stride, int dimension, bool update, int togo_dimension){
    for(int i = 0; i < dimension; i++){
        elem_t sum = 0;
        for(int j = 0; j <= i; j++){
            if(j < i)
                sum += *(L+L_stride*i+j) * (*(x+j));
            else {
                elem_t Lx = *(b+j) - sum;
                *(x+j) = Lx / (*(L+L_stride*i+j));
            }
        }
    }
    if(update && togo_dimension > 0){
        for(int i = dimension; i < togo_dimension; i++){
            elem_t sum = 0;
            for(int j = 0; j <= dimension; j++){
                if(j < i)
                    sum += *(L+L_stride*i+j) * (*(x+j));
                else {
                    elem_t Lx = *(b+j) - sum;
                    *(x+j) = Lx / (*(L+L_stride*i+j));
                }
            }
        }       
    }
}

void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*1000))%1000);
    printf("\n");
  }
}

int full_is_equal(int stride, elem_t x[stride][stride], elem_t y[stride][stride], int dimension) {
  for (size_t i = 0; i < dimension; ++i)
    for (size_t j = 0; j < dimension; ++j)
      if (((int)(x[i][j]*100)) != ((int)(y[i][j]*100))){
		  printf("i: %d, j: %d, value: %d.%d, %d.%d \n", i, j, (int)x[i][j], ((int)(x[i][j]*1000))%1000 , (int)y[i][j], ((int)(y[i][j]*1000))%1000);
         //return 0;
	  }
  return 1;
}



void lower_triangle_inverse(elem_t* A, int stride, int dim, elem_t M[dim][dim]){
	for(int i = 0; i < dim; i++)
		for(int j = 0; j < dim; j++)
			M[i][j] = 0;

	for(int i = 0; i < dim; i++){
		M[i][i] = 1/(*(A+i*stride+i));
		for(int j = i+1; j < dim; j++){
			elem_t sum = 0;
            int jstride = j * stride;
			for(int k = i; k < j; k++)
				sum += (*(A+jstride+k))*M[k][i];///(*(A+j*stride+j));
			M[j][i] = M[j][i] - (sum)/(*(A+jstride+j));
		}
	}
}


void full_left_chol(elem_t* L, int block_dim, int stride){
	for(int j=0; j < block_dim; j++){
        int jstride = j*stride;
		for(int k=0; k < j; k++){
			*(L+jstride+j) -= (*(L+jstride+k))*(*(L+jstride+k));
			for(int i = j+1; i < block_dim; i++)
				*(L+i*stride+j) -= (*(L+i*stride+k))*(*(L+jstride+k));
		}
		*(L+jstride+j) = (sqrt(*(L+jstride+j)));
		for(int i = j+1; i < block_dim; i++)
			*(L+i*stride+j) = ((*(L+i*stride+j))/(*(L+jstride+j)));
	    for(int i = 0; i < j; i++)
            *(L+i*stride+j) = 0;
    }
}

void copy_matrix(int block_size, int A_dim, int B_dim, elem_t* A, elem_t* B){
	for(int i = 0; i < block_size; i++)
		for(int j = 0; j < block_size; j++)
			*(B+i*B_dim+j) = *(A+i*A_dim+j);
}

elem_t temp_inv[BLOCK_DIM][BLOCK_DIM] = {0};
void block_left_chol(elem_t* L, int dimension, int stride, int block_size){
	int num_block = dimension / block_size;
    int num_array = 1;
    for(int k = 0; k < num_block; k++){
		if(k > 0) full_transposed_matmul(1, 1, k, block_size, stride, stride, stride, L+block_size*(k*stride), L+block_size*(k*stride), L+block_size*(k*stride+k), true, num_array);	
		full_left_chol(L+block_size*(k*stride+k), block_size, stride);
		//elem_t temp_inv[block_size][block_size] = {0};
		lower_triangle_inverse(L+(k*stride+k)*block_size, stride, block_size, temp_inv);

        if(k > 0 && k < num_block - 1) full_transposed_matmul(num_block-k-1, 1, k, block_size, stride, stride, stride, L+block_size*((k+1)*stride), L+block_size*(k*stride), L+block_size*((k+1)*stride+k), true, num_array);

        if (k < num_block - 1) 
          full_transposed_matmul(num_block-k-1, 1, 1, block_size, stride, block_size, stride, L+block_size*((k+1)*stride+k), (elem_t*) temp_inv, L+block_size*((k+1)*stride+k), false, num_array);
	}
}
/*
void block_left_chol(elem_t* L){
	for(int k = 0; k < num_block; k++){
		if(k > 0) full_transposed_matmul(1, 1, k, MAT_DIM, MAT_DIM, MAT_DIM, L+block_dim*(k*MAT_DIM), L+block_dim*(k*MAT_DIM), L+block_dim*(k*MAT_DIM+k), true);	
		full_left_chol(L+block_dim*(MAT_DIM*k+k));
		elem_t temp_inv[block_dim][block_dim] = {0};
		lower_triangle_inverse(L+(k*MAT_DIM+k)*block_dim, temp_inv);

		for(int i = k+1; i < num_block; i++){
			if(k > 0) full_transposed_matmul(1, 1, k, MAT_DIM, MAT_DIM, MAT_DIM, L+block_dim*(i*MAT_DIM), L+block_dim*(k*MAT_DIM), L+block_dim*(i*MAT_DIM+k), true);
		}
//			for(int j = 0; j < k; j++)
//				full_transposed_matmul(block_dim, MAT_DIM, MAT_DIM, MAT_DIM, L+block_dim*(i*MAT_DIM+j), L+block_dim*(k*MAT_DIM+j), L+block_dim*(i*MAT_DIM+k), true);	
		for(int i = k+1; i < num_block; i++){		
			full_transposed_matmul(1, 1, 1, MAT_DIM, block_dim, MAT_DIM, L+block_dim*(i*MAT_DIM+k), (elem_t*)temp_inv, L+block_dim*(i*MAT_DIM+k), false);
		}
	}
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
    int num_block = NUM_BLOCK;
    int block_size = BLOCK_DIM;
    static elem_t LR_block[MAT_DIM][MAT_DIM] row_align(1) = {0};
	static elem_t LL_block[MAT_DIM][MAT_DIM] row_align(1) = {0};
	unsigned long cpu_start = 0;
	unsigned long cpu_end = 0;


#if CHECK_RESULT == 1

	 for(int i = 0; i < num_block; i++)
		 for(int j = 0; j < num_block; j++)
			 for(int ii = 0; ii < block_size; ii++){
				 for(int jj = 0; jj < block_size; jj++){
					 if(j > i){
	//					 LR_block[i*block_size+ii][j*block_size+jj] = 0;
						 LL_block[i*block_size+ii][j*block_size+jj] = 0;
					 }
					 else{
   //					 LR_block[i*block_size+ii][j*block_size+jj] = in_A[i*block_size+ii][j*block_size+jj];
					    LL_block[i*block_size+ii][j*block_size+jj] = in_A[i*block_size+ii][j*block_size+jj];
					 }
				 }
			 }

#endif
    printf("Starting block left CPU chol\n");
    cpu_start = read_cycles();
    block_left_chol((elem_t *) LL_block, MAT_DIM, MAT_DIM_S, block_size);
    cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);




#if CHECK_RESULT == 1
/*
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
*/	
	 if (!full_is_equal(MAT_DIM_S, LL_block, gold_L, MAT_DIM)) {
      printf("C:\n");
    //  full_printMatrix(LL_block);
      printf("Block left Gold:\n");
    //  full_printMatrix(gold_L);
      printf("\n");
      exit(1);
    }
	 printf("correct \n");
#endif

  exit(0);
}

