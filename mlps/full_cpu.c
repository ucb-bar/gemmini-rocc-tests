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
#include "full_data.h"
#include "functions.h"

#define CHECK_RESULT 1
/*
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif
*/

// temp matrix for CA^-1
static elem_t temp[D_DIM][A_DIM_S] = {0};

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
    // input: A, output: A_inv_cpu
    cpu_A_inv(A_DIM, A_DIM_S, A, A_inv_cpu, A_BLOCK_DIM);
    printf("compare A inverse\n");
    full_is_equal(A_DIM, A_DIM, A_inv_gold, A_inv_cpu);
    printf("Starting CPU Schur\n");
    // input: A_inv_cpu, C, output: D
    schur(A_DIM, A_DIM_S, D_DIM, D_DIM_S, A_DIM_S, (elem_t*) A_inv_cpu, (elem_t*) C, (elem_t*) D, (elem_t*) temp, A_BLOCK_DIM); 
    cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    printf("compare schur\n");
    full_is_equal(D_DIM_S, D_DIM_S, S, D);

    elem_t* D_pointer = (elem_t*) D;
    block_left_chol(D_pointer, D_DIM, D_DIM_S, BLOCK_DIM);

    static elem_t dx_cpu[D_DIM] = {0};
    for(int i = 0; i < NUM_BLOCK; i++){
        gaussian_elimination(D_pointer, (elem_t*) dx_cpu + BLOCK_DIM * i, (elem_t*) b_vec + BLOCK_DIM * i, D_DIM_S, BLOCK_DIM, true, D_DIM - BLOCK_DIM * (i+1));
        D_pointer += BLOCK_DIM * D_DIM_S + BLOCK_DIM;
    }

    vec_is_equal(dx_gold, dx_cpu, D_DIM);
    exit(0);
}

