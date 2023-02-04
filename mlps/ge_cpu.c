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
#include "ge_data.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#define NUM_BLOCK 4
#define BLOCK_DIM ((int)(MAT_DIM / NUM_BLOCK))
/*
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif
*/

// Lx = b (solve x)
// for partial matrix, update is whether to update the below part of b (togo_dimension)
void cpu_gaussian_elimination(elem_t* L, elem_t* x, elem_t* b, int L_stride, int dimension, bool update, int togo_dimension){
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
        for(int i = dimension; i < dimension+togo_dimension; i++){
            elem_t sum = 0;
            for(int j = 0; j <= dimension; j++){
                if(j < dimension)
                    sum += *(L+(L_stride*i+j)) * (*(x+j));
                else {
                    elem_t Lx = *(b+i) - sum;
                    //printf("i: %d, Lx x 1000: %d, L x 1000: %d\n", i, (int)(1000*Lx), (int)(1000 * (*(L+(L_stride*i+j)))));
                    *(b+i) -= sum;
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

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            if (((int)(a[i]*100)) != ((int)(b[i]*100)))
			    printf("i: %d, value: %d.%d, %d.%d \n", i, (int)a[i], ((int)(a[i]*1000))%1000 , (int)b[i], ((int)(b[i]*1000))%1000);
            //return false;
    return true;
}
void copy_matrix(int block_size, int A_dim, int B_dim, elem_t* A, elem_t* B){
	for(int i = 0; i < block_size; i++)
		for(int j = 0; j < block_size; j++)
			*(B+i*B_dim+j) = *(A+i*A_dim+j);
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

    static elem_t dx_cpu[MAT_DIM] = {0};
    printf("Starting CPU back propagation\n");
    cpu_start = read_cycles();
    elem_t* A_pointer = (elem_t*) A;
    for(int i = 0; i < NUM_BLOCK; i++){
        cpu_gaussian_elimination(A_pointer, (elem_t*) dx_cpu + BLOCK_DIM * i, (elem_t*) b_vec + BLOCK_DIM * i, MAT_DIM_S, BLOCK_DIM, true, MAT_DIM - BLOCK_DIM * (i+1));
        A_pointer += BLOCK_DIM * MAT_DIM_S + BLOCK_DIM;
    }
    cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);


    vec_is_equal(dx_gold, dx_cpu, MAT_DIM);
    exit(0);
}

