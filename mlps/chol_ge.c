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

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            //if (((int)(a[i]*100)) != ((int)(b[i]*100)))
            if(abs_diff((int)(a[i]*100), (int)(b[i]*100)) > 1)
			    printf("i: %d, value: %d.%d, %d.%d \n", i, (int)a[i], ((int)(a[i]*1000))%1000 , (int)b[i], ((int)(b[i]*1000))%1000);
            //return false;
    return true;
}

void full_transposed_matmul(int I_block, int J_block, int K_block, int block_dim, int A_stride, int B_stride, int C_stride, elem_t* A, elem_t* B, elem_t* C, bool sub, int num_array) {
/*	
	tiled_opcode_matmul_auto_multi(block_dim*I_block, block_dim*J_block, block_dim*K_block,
			A_stride, B_stride, C_stride, C_stride,
            false, false, false, false,
			A, B, sub ? C : NULL, C,
			MVIN_SCALE_IDENTITY, sub ? (-1) :MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            false, true,
			NO_ACTIVATION, ACC_SCALE_IDENTITY, false, false, false, 
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
void gaussian_elimination(elem_t* L, elem_t* x, elem_t* b, int L_stride, int dimension, bool update, int togo_dimension){
    for(int i = 0; i < dimension; i++){
        elem_t sum = 0;
        for(int j = 0; j <= i; j++){
            if(j < i)
                sum += *(L+(L_stride*i+j)) * (*(x+j));
            else {
                elem_t Lx = *(b+j) - sum;
                //printf("i: %d, Lx x 1000: %d, L x 1000: %d\n", i, (int)(1000*Lx), (int)(1000 * (*(L+(L_stride*i+j)))));
                *(x+j) = Lx / (*(L+(L_stride*i+j)));
            }
        }
    }
    
    if(update && togo_dimension > 0){
       tiled_matmul_auto(1, togo_dimension, dimension, x, L+L_stride*dimension, b+dimension, b+dimension,
               L_stride, L_stride, L_stride, L_stride,
               false, false, false, false,
               -1, 1, 1,
               0, 1, 0, false,
               false, true, // transpose
               false, false,
               3, WS);
    }
    /*
    if(update && togo_dimension > 0){
       tiled_matmul_auto(togo_dimension, 1, dimension, L+L_stride*dimension, x, b+dimension, b+dimension,
               L_stride, 1, 1, 1,
               false, false, false, false,
               1, -1, 1,
               0, 1, 0, false,
               false, false, // transpose
               false, false,
               3, WS);
    }
    */
}
void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*1000))%1000);
    printf("\n");
  }
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
		//printf("k: %d\n", k);
        if(k > 0) full_transposed_matmul(1, 1, k, block_size, stride, stride, stride, L+block_size*(k*stride), L+block_size*(k*stride), L+block_size*(k*stride+k), true, num_array);	
		//printf("left looking chol \n");
        full_left_chol(L+block_size*(k*stride+k), block_size, stride); 
        //printf("cpu inversion\n");
		lower_triangle_inverse(L+(k*stride+k)*block_size, stride, block_size, temp_inv);

        //printf("transposed matmul\n");
        if(k > 0 && k < num_block - 1) full_transposed_matmul(num_block-k-1, 1, k, block_size, stride, stride, stride, L+block_size*((k+1)*stride), L+block_size*(k*stride), L+block_size*((k+1)*stride+k), true, num_array);

        //printf("transposed matmul\n");
        if (k < num_block - 1) 
          full_transposed_matmul(num_block-k-1, 1, 1, block_size, stride, block_size, stride, L+block_size*((k+1)*stride+k), (elem_t*) temp_inv, L+block_size*((k+1)*stride+k), false, num_array);
	}
}
	

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
	unsigned long cpu_start = 0;
	unsigned long cpu_end = 0;

    printf("Starting block left CPU chol\n");
    cpu_start = read_cycles();
    block_left_chol((elem_t *) in_A, MAT_DIM, MAT_DIM_S, block_size);
    cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);

    static elem_t dx_cpu[MAT_DIM] = {0};
    printf("Starting CPU back propagation\n");
    elem_t* A_pointer = (elem_t*) in_A;
    for(int i = 0; i < NUM_BLOCK; i++){
        gaussian_elimination(A_pointer, (elem_t*) dx_cpu + BLOCK_DIM * i, (elem_t*) b_vec + BLOCK_DIM * i, MAT_DIM_S, BLOCK_DIM, true, MAT_DIM - BLOCK_DIM * (i+1));
        A_pointer += BLOCK_DIM * MAT_DIM_S + BLOCK_DIM;
    }

#if CHECK_RESULT == 1
    vec_is_equal(dx_gold, dx_cpu, MAT_DIM);
	 printf("correct \n");
#endif

  exit(0);
}

