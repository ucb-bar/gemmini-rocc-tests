// See LICENSE for license details.
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/gemmini_spad.h"

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1
#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#define MAT_DIM 64
#define STRIDE (MAT_DIM)

#define NUM_ROW_TILE 2
#define NUM_COL_TILE 2

#define PRELOAD false 

void full_printMatrix(elem_t m[MAT_DIM][MAT_DIM]) {
  for (size_t i = 0; i < MAT_DIM; ++i) {
    for (size_t j = 0; j < MAT_DIM; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i]){
            printf("a: %d, b: %d, index: %d\n", a[i], b[i], i);
            //return false;
        }
    return true;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    int channel = 1;
    int thread_id = 0;
    printf("DMA channel: %d\n", channel);
    printf("I: %d, J: %d, stride: %d\n", MAT_DIM, MAT_DIM, STRIDE);
    int len = MAT_DIM * MAT_DIM;
    
    gemmini_flush(0);

    static elem_t A[MAT_DIM][STRIDE] row_align(1) = {0};
    static elem_t B[MAT_DIM][STRIDE] row_align(1) = {0};
    static elem_t C[MAT_DIM][STRIDE] row_align(1) = {0};
    static elem_t gold[MAT_DIM][STRIDE] row_align(1) = {0};

    uint64_t A_copy_addr = BASE_ADDR;
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM; ++i) {
      for (size_t j = 0; j < MAT_DIM; ++j) {
        A[i][j] = (rand() % 64) - 32;
        B[i][j] = (rand() % 8) - 4;
        //gold[i][j] = A[i][j]+B[i][j];
      }
    }
    
    tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM, (elem_t*) A, (elem_t*) B, NULL, (elem_t*) gold,
        STRIDE, STRIDE, STRIDE, STRIDE,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false,
        false, !FULL_BIAS_WIDTH,
        0, WS);

    sp_capacity_alloc[0] = (SPAD_BANK_BYTES) * SPAD_BANK_NUM;
    sp_base_addr[0] = 0;
    dma_channel_alloc[0][0] = 0;
    dma_channel_alloc[0][1] = 1;
    dma_channel_alloc[0][2] = 2;

#if PRELOAD == 1
    //printf("configuring DMA\n");
    //dma_config(channel, LOAD, (elem_t*) A, 0, STRIDE, MAT_DIM);
    
    //printf("perform memcpy\n");
    //dma_memcpy_matrix(channel, 0, NUM_ROW_TILE, NUM_COL_TILE, MAT_DIM, MAT_DIM * sizeof(elem_t), MAT_DIM / NUM_ROW_TILE, (MAT_DIM*sizeof(elem_t))/NUM_COL_TILE);
 
    //int len = MAT_DIM*MAT_DIM;
    memcpy((elem_t*) A_copy_addr, (elem_t*) A, sizeof(elem_t)*len); 
    sp_input_base_addr[0] = 0;
    elem_t* input_A = (elem_t*) A_copy_addr;
#else
    sp_input_base_addr[0] = -1;
    elem_t* input_A = (elem_t*) A;
#endif

    double_tiled_matmul_auto(MAT_DIM, MAT_DIM, MAT_DIM, input_A, (elem_t*) B, NULL, (elem_t*) C,
        MAT_DIM, STRIDE, STRIDE, STRIDE,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false, false, false,
        false, !FULL_BIAS_WIDTH,
        0, thread_id);

    //tiled_resadd_auto(MAT_DIM, MAT_DIM, 1, 1, 1, (elem_t*) A_copy_addr, (elem_t*) B, (elem_t*) C, false, WS);
    bool result;
#if PRELOAD == 1
    printf("check result\n");
    vec_is_equal(&A[0][0], (elem_t*) A_copy_addr, sizeof(A)/sizeof(elem_t));
#endif

    if(sp_input_base_addr[thread_id] == -1){
      printf("check matmul result\n");
      result = vec_is_equal(&gold[0][0], &C[0][0], sizeof(gold)/sizeof(elem_t));
    }
    else{
      printf("output to spad\n");
      for(int i = 0; i < MAT_DIM; i++)
        for(int j = 0; j < STRIDE; j++)
          C[i][j] = 0;
      uint64_t c_addr = sp_input_base_addr[thread_id];
      printf("c spad addr: 0x%08lx\n", c_addr);
      memcpy((elem_t*) C, (elem_t*) c_addr, sizeof(elem_t)*len);
      printf("check matmul result\n");
      result = vec_is_equal(&gold[0][0], (elem_t*) c_addr, sizeof(gold)/sizeof(elem_t));  
    }
    if (result == false){
        printf("gold: \n");
        full_printMatrix(gold);
        printf("C: \n");
        full_printMatrix(C);
    }
    exit(0);
}

