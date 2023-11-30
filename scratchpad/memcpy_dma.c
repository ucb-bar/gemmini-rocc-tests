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
#define MAT_DIM_I 4
#define MAT_DIM_J 48
#endif

//#define MAT_DIM 32
#define NUM_ROW_TILE 2
#define NUM_COL_TILE 2

#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU true

#define PageSize 4096

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

    dma_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
    static elem_t B[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
    static elem_t gold[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));
    printf("A copy addr: 0x%08lx\n", A_copy_addr);
    printf("A source addr: 0x%08lx\n", A);

    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 64) - 32;
        //B[i][j] = (rand() % 8) - 4;
        //gold[i][j] = A[i][j]+B[i][j];
      }
    }
    printf("configuring DMA\n");
    dma_source_config(channel, (elem_t*) A, MAT_DIM_J);
    dma_dest_config(channel, A_copy_addr, MAT_DIM_J);

    printf("perform memcpy\n");
    bool granted = false;
    int index = 0;
    dma_memcpy_tile(channel, granted, 0, 0, index, MAT_DIM_I, (MAT_DIM_J) * sizeof(elem_t));
    
    //tiled_resadd_auto(MAT_DIM_I, MAT_DIM_J, 1, 1, 1, (elem_t*) A_copy_addr, (elem_t*) B, (elem_t*) C, false, WS);
    printf("granted %d, check result\n", granted);
    vec_is_equal(&A[0][0], (elem_t*) A_copy_addr, sizeof(A)/sizeof(elem_t));
    //vec_is_equal(&gold[0][0], &C[0][0], sizeof(gold)/sizeof(elem_t));

    exit(0);
}

