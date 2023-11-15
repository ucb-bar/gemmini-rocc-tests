// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#define SIZE 2

#include "include/gemmini_testutils.h"
#include "include/rerocc.h"
#include "include/dma.h"

#define ACTIVATION NO_ACTIVATION

#define NO_BIAS 1
#define REPEATING_BIAS 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0

#ifndef BAREMETAL

#define MAT_DIM_I 128
#define MAT_DIM_K 512
#define MAT_DIM_J 256

#else

#define MAT_DIM_I 196
#define MAT_DIM_K 256
#define MAT_DIM_J 1024

#endif

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif

#define PageSize 4096

#define ACC_ID 3

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
    // spica in tile 4, gemmini in tile 1
    rr_acquire_single(0, 4);
    rr_acquire_single(1, ACC_ID);

    rr_set_opc(XCUSTOM_ACC, 1);
    gemmini_flush(0);
    rr_set_opc(XCUSTOM_DMA, 0);
    dma_flush(0);

#if A_TRANSPOSE==0
    static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#else
    static elem_t A[MAT_DIM_K][MAT_DIM_I] row_align(MAX_BLOCK_LEN);
#endif

#if B_TRANSPOSE==0
    static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
#else
    static elem_t B[MAT_DIM_J][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
#endif

    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static acc_t D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

    uint64_t A_copy_addr = (BASE_ADDR & ~(PageSize-1));
    printf("A copy addr: 0x%08lx\n", A_copy_addr);
    printf("A source addr: 0x%08lx\n", A);
    uint64_t B_copy_addr = A_copy_addr + (MAT_DIM_I * MAT_DIM_K) * sizeof(elem_t);// + 64*3;
    uint64_t C_copy_addr = B_copy_addr + (MAT_DIM_K * MAT_DIM_J) * sizeof(elem_t);// + 64*3;
    uint64_t D_copy_addr = C_copy_addr + (MAT_DIM_I * MAT_DIM_J) * sizeof(elem_t);
/*
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A[i][j] = (rand() % 64) - 32;
        //B[i][j] = (rand() % 8) - 4;
        //gold[i][j] = A[i][j]+B[i][j];
      }
    }
    */
    printf("configuring DMA\n");
    dma_source_config(0, (elem_t*) A, MAT_DIM_K);
    dma_dest_config(0, A_copy_addr, MAT_DIM_K);
    dma_source_config(1, (elem_t*) B, MAT_DIM_J);
    dma_dest_config(1, B_copy_addr, MAT_DIM_J);
    dma_source_config(2, (elem_t*) D, MAT_DIM_J * sizeof(acc_t));
    dma_dest_config(2, D_copy_addr, MAT_DIM_J * sizeof(acc_t));

    printf("perform memcpy\n");
    bool granted = false;
    int index = 0;
    printf("copy A\n");
    //dma_memcpy_tile(0, granted, 0, 0, index, MAT_DIM_I, (MAT_DIM_K) * sizeof(elem_t));
    printf("copy B\n");
    //dma_memcpy_tile(1, granted, 0, 0, index, MAT_DIM_K, (MAT_DIM_J) * sizeof(elem_t));
    printf("copy D\n");
    //dma_memcpy_tile(2, granted, 0, 0, index, 1, (MAT_DIM_J) * sizeof(acc_t));

    rr_fence(0);
    printf("Starting gemmini matmul\n");
    printf("gemmini spad rows: %d, acc rows: %d \n", BANK_ROWS, ACC_ROWS);
    printf("I: %d, J: %d, K: %d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("NO_BIAS: %d, REPEATING_BIAS: %d\n", NO_BIAS, REPEATING_BIAS);
    printf("A_TRANSPOSE: %d, B_TRANSPOSE: %d\n", A_TRANSPOSE, B_TRANSPOSE);

    rr_set_opc(XCUSTOM_ACC, 1);
    uint64_t start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
        (elem_t*) A_copy_addr, (elem_t*) B_copy_addr, NO_BIAS ? NULL : (elem_t*) D_copy_addr, (elem_t*) C_copy_addr,
        A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
        A_TRANSPOSE, B_TRANSPOSE,
        false, false,
        0,
        WS);

    rr_fence(1);

    uint64_t end = read_cycles();
    printf("Cycles taken: %llu\n", end-start);

    const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const uint64_t utilization = 100 * ideal_cycles / (end-start);
    printf("Total macs: %llu\n", total_macs);
    printf("Ideal cycles: %llu\n", ideal_cycles);
    printf("Utilization: %llu%%\n", utilization);

    //ToDo: check matmul result
    // release gemmini
    rr_release(0);
    rr_release(1);
    exit(0);
}

