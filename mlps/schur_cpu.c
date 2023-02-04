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
#include "schur_data.h"

#define CHECK_RESULT 1
#define BLOCK_DIM 3 
/*
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif
*/

void full_is_equal(int dim_r, int dim_c, elem_t x[dim_r][dim_c], elem_t y[dim_r][dim_c]) {
  for (size_t i = 0; i < dim_r; ++i)
    for (size_t j = 0; j < dim_c; ++j)
      if (abs_diff((int)(x[i][j]*1000), (int)(y[i][j]*1000)) > 1){
          printf("i: %d, j: %d, x: %d, y: %d\n", i, j, (int)(x[i][j]*1000), (int)(y[i][j]*1000));
      }
}

// there should be better way to store matrix
void cpu_A_inv(int mat_dim, int stride, elem_t in[mat_dim][stride], elem_t out[mat_dim][stride], int block_dim){

    int num_block = ceil_divide_int(mat_dim, block_dim);
    for (int i = 0; i < num_block; i++){
      int start_index = i * block_dim;
      if(i!=num_block - 1|| mat_dim%block_dim == 0){
        elem_t mul1122 = in[start_index+1][start_index+1]*in[start_index+2][start_index+2];
        elem_t mul2112 = in[start_index+2][start_index+1]*in[start_index+1][start_index+2];
        elem_t mul1022 = in[start_index+1][start_index]*in[start_index+2][start_index+2];
        elem_t mul1220 = in[start_index+1][start_index+2]*in[start_index+2][start_index];
        elem_t mul1021 = in[start_index+1][start_index]*in[start_index+2][start_index+1];
        elem_t mul1120 = in[start_index+1][start_index+1]*in[start_index+2][start_index];
        elem_t det = in[start_index][start_index]*(mul1122-mul2112) - in[start_index][start_index+1]*(mul1022-mul1220) + in[start_index][start_index+2]*(mul1021-mul1120);
        elem_t invdet = 1/det;

        out[start_index][start_index] = (mul1122 - mul2112) * invdet;
        out[start_index][start_index+1] = (in[start_index][start_index+2]*in[start_index+2][start_index+1]-in[start_index][start_index+1]*in[start_index+2][start_index+2])*invdet;
        out[start_index][start_index+2] = (in[start_index][start_index+1]*in[start_index+1][start_index+2]-in[start_index][start_index+2]*in[start_index+1][start_index+1])*invdet;
        out[start_index+1][start_index] = (mul1220 - mul1022) * invdet;
        out[start_index+1][start_index+1] = (in[start_index][start_index]*in[start_index+2][start_index+2]-in[start_index][start_index+2]*in[start_index+2][start_index])*invdet;
        out[start_index+1][start_index+2] = (in[start_index+1][start_index]*in[start_index][start_index+2]-in[start_index][start_index]*in[start_index+1][start_index+2])*invdet;
        out[start_index+2][start_index] = (mul1021 - mul1120) * invdet;
        out[start_index+2][start_index+1] = (in[start_index+2][start_index]*in[start_index][start_index+1]-in[start_index][start_index]*in[start_index+2][start_index+1])*invdet;
        out[start_index+2][start_index+2] = (in[start_index][start_index]*in[start_index+1][start_index+1]-in[start_index+1][start_index]*in[start_index][start_index+1])*invdet;
      }
      else if(mat_dim%block_dim == 2){
        elem_t d = in[start_index][start_index]*in[start_index+1][start_index+1] - in[start_index][start_index+1]*in[start_index+1][start_index];
        out[start_index][start_index] = in[start_index+1][start_index+1]/d;
        out[start_index+1][start_index] = in[start_index][start_index]/d;
        out[start_index][start_index+1] = (-1)*in[start_index][start_index+1]/d;
        out[start_index+1][start_index] = (-1)*in[start_index+1][start_index]/d;
      }
      else{
        out[start_index][start_index] = 1/in[start_index][start_index];
      }
    }
}

void cpu_matmul(int out_row, int out_col, int a_stride, int b_row, int b_stride, int d_stride, int out_stride, elem_t in_a[out_row][a_stride], elem_t in_b[b_row][b_stride], elem_t bias[out_row][d_stride], elem_t out[out_row][out_stride]) {
  for (size_t r = 0; r < out_row; r++)
    for (size_t c = 0; c < out_col; c++) {
      out[r][c] = d_stride > 0 ? bias[r][c] : 0;
      for (size_t k = 0; k < b_row; k++)
        out[r][c] += in_a[r][k]*in_b[k][c];
    }
}
void cpu_matmul_transpose(int out_row, int out_col, int a_stride, int K, int b_stride, int d_stride, int out_stride, elem_t in_a[out_row][a_stride], elem_t in_b[out_col][b_stride], elem_t bias[out_row][d_stride], elem_t out[out_row][out_stride]) {
  for (size_t r = 0; r < out_row; r++)
    for (size_t c = 0; c < out_col; c++) {
      out[r][c] = d_stride > 0 ? bias[r][c] : 0;
      for (size_t k = 0; k < K; k++)
        out[r][c] += in_a[r][k]*in_b[c][k];
    }
}

static elem_t temp_out[D_DIM][A_DIM] = {0};
// D-CA^-1B = D-CA^-1C^T
// for now, input already inversed
void cpu_schur(int a_dim, int a_stride, int d_dim, int d_stride, int c_stride, elem_t* A_inv_in, elem_t* in_C, elem_t* in_D, int block_dim){// elem_t A_inv_in[a_dim][a_stride], elem_t in_C[d_dim][c_stride], elem_t in_D[d_dim][d_stride], int block_dim){
    int num_block = ceil_divide_int(a_dim, block_dim);
    //elem_t temp_out[d_dim][a_stride] = {0};
    elem_t* temp_out_pt = (elem_t*) temp_out;
    
    for (int i = 0; i < num_block; i++){
      int start_index = i * block_dim;
      int eff_block_dim = i != num_block - 1 ? block_dim : (a_dim - start_index);
      elem_t* in_A = in_C + start_index;
      elem_t* in_B = A_inv_in + a_stride * start_index + start_index;
      elem_t* out = temp_out_pt + start_index;
      tiled_matmul_auto(d_dim, eff_block_dim, eff_block_dim, in_A, in_B, NULL, out,
            c_stride, a_stride, d_stride, a_stride,
            false, false, false, false,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false, false, 0,
            0, WS);
    }
    /*
    tiled_matmul_auto(d_dim, a_dim, a_dim, (elem_t*) in_C, (elem_t*) A_inv_in, NULL, temp_out_pt,
        c_stride, a_stride, d_stride, a_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, false, false, 0,
        0, WS);
        */
    printf("finished CA^-1\n");
    
    tiled_matmul_auto(d_dim, d_dim, a_dim, temp_out_pt,  (elem_t*) in_C, (elem_t*) in_D, (elem_t*) in_D,
        a_stride, c_stride, d_stride, d_stride,
        false, false, false, false,
        MVIN_SCALE_IDENTITY, -1, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
        false, true, false, 1,
        0, WS);
}

 
void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
		 printf("%d.%d ", (int)m[i][j], ((int)(m[i][j]*1000))%1000);
    printf("\n");
  }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            if (((int)(a[i]*100)) != ((int)(b[i]*100)))
			    printf("i: %d, value: %d.%d, %d.%d \n", i, (int)a[i], ((int)(a[i]*1000))%1000 , (int)b[i], ((int)(b[i]*1000))%1000);
            //return false;
    return true;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);
	unsigned long cpu_start = 0;
	unsigned long cpu_end = 0;

    static elem_t A_inv_cpu[A_DIM][A_DIM] = {0};
    printf("Starting CPU A inverse\n");
    cpu_start = read_cycles();
    //elem_t* A_pointer = (elem_t*) A;
    cpu_A_inv(A_DIM, A_DIM, A, A_inv_cpu, BLOCK_DIM);
    printf("Starting CPU Schur\n");
    cpu_schur(A_DIM, A_DIM, D_DIM, D_DIM, A_DIM, (elem_t*) A_inv_cpu, (elem_t*) C, (elem_t*) D, BLOCK_DIM); 
    cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);

    printf("compare A inverse\n");
    full_is_equal(A_DIM, A_DIM, A_inv_gold, A_inv_cpu);
    printf("compare schur\n");
    full_is_equal(D_DIM, A_DIM, CAinv, temp_out);
    full_is_equal(D_DIM, D_DIM, S, D);
    exit(0);
}

