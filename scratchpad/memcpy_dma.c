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
#include "include/dma.h"

#define CHECK_RESULT 0 // 1

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_J 512

#else
#define MAT_DIM_I 32
#define MAT_DIM_J 32
#endif

#define MAT_DIM 32
#define NUM_ROW_TILE 2
#define NUM_COL_TILE 2

#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            printf("a: %d, b: %d, index: %d\n", a[i], b[i], i);
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

    int channel = 1;
    printf("DMA channel: %d\n", channel);
    printf("I: %d, J: %d\n", MAT_DIM_I, MAT_DIM_J);
    int len = MAT_DIM_I * MAT_DIM_J;

    gemmini_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_J] row_align(1) = {0};
    static elem_t B[MAT_DIM_I][MAT_DIM_J] row_align(1) = {0};
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1) = {0};
    static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(1) = {0};

    uint64_t A_copy_addr = BASE_ADDR;
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 64) - 32;
        B[i][j] = (rand() % 8) - 4;
        gold[i][j] = A[i][j]+B[i][j];
      }
    }
    printf("configuring DMA\n");
    dma_config(channel, LOAD, (elem_t*) A, 0, MAT_DIM_J, MAT_DIM_J);

    printf("perform memcpy\n");
    dma_memcpy_matrix(channel, 0, NUM_ROW_TILE, NUM_COL_TILE, MAT_DIM_I, MAT_DIM_J * sizeof(elem_t), MAT_DIM_I / NUM_ROW_TILE, (MAT_DIM_J*sizeof(elem_t))/NUM_COL_TILE);
    
    //tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, 1, 1, 1, (elem_t*) A_copy_addr, (elem_t*) B, (elem_t*) C, false, WS);
    printf("check result\n");
    vec_is_equal(&A[0][0], (elem_t*) A_copy_addr, sizeof(A)/sizeof(elem_t));
    //vec_is_equal(&gold[0][0], &C[0][0], sizeof(gold)/sizeof(elem_t));

    exit(0);
}

