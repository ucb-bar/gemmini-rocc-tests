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

#define BATCH_SIZE 1
#define IN_ROW_DIM 14
#define IN_COL_DIM 14
#define IN_CHANNELS 64
#define OUT_CHANNELS 64
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1
 
#define OUT_ROW_DIM ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

#define PRELOAD true

#define IN_STRIDE (IN_CHANNELS)
#define WEIGHT_STRIDE (OUT_CHANNELS)
#define OUT_STRIDE (OUT_CHANNELS)


bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i]){
            //printf("a: %d, b: %d, index: %d\n", a[i], b[i], i);
            return false;
        }
    return true;
}

void init_random(elem_t * buf, int row, int col, int stride){
    elem_t i = 0;
    for(int r = 0; r < row; r++){
        for(int c = 0; c < col; c++){
            elem_t * ptr = buf + r * stride + c;
            *ptr = (rand() % 5) - 2;
        }
    }
}


void init_one(elem_t * buf, int row, int col, int stride){
    elem_t i = 0;
    for(int r = 0; r < row; r++){
        for(int c = 0; c < col; c++){
            elem_t * ptr = buf + r * stride + c;
            *ptr = 1;
        }
    }
}

void init_random_acc(acc_t * buf, int len) {
    elem_t i = 0;
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
      *ptr = (rand() % 5) - 2;
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
    }
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
    
    gemmini_flush(0);

    uint64_t A_copy_addr = BASE_ADDR;
    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);

    static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_STRIDE];
    static acc_t bias[OUT_CHANNELS];
    static elem_t weights_mat[PATCH_SIZE][OUT_STRIDE];
    static elem_t output_mat[N_PATCHES][OUT_STRIDE] = {0};
    static elem_t output_mat_spad[N_PATCHES][OUT_STRIDE] = {0};

    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], BATCH_SIZE*IN_ROW_DIM*IN_COL_DIM, IN_CHANNELS, IN_STRIDE);

    printf("Randomize weights...\n");
    init_random(&weights_mat[0][0], PATCH_SIZE, OUT_CHANNELS, OUT_STRIDE);

    printf("Randomize bias...\n");
    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    else
        init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));

    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_stride_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        IN_STRIDE, WEIGHT_STRIDE, OUT_STRIDE,
        false, false, false, false, false,

        (elem_t*)input,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

        WS);
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);


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
 
    int len = BATCH_SIZE*IN_ROW_DIM*IN_COL_DIM*IN_STRIDE;
    memcpy((elem_t*) A_copy_addr, (elem_t*) input, sizeof(elem_t)*len); 
    sp_input_base_addr[0] = 0;
    elem_t* input_A = (elem_t*) A_copy_addr;
#else
    sp_input_base_addr[0] = -1;
    elem_t* input_A = (elem_t*) input;
#endif

    double_tiled_conv_auto(
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        IN_STRIDE, WEIGHT_STRIDE, OUT_STRIDE,
        false, false, false, false, false,

        input_A,
        (elem_t*)weights_mat,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output_mat_spad,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

        thread_id);
    //tiled_resadd_auto(MAT_DIM, MAT_DIM, 1, 1, 1, (elem_t*) A_copy_addr, (elem_t*) B, (elem_t*) C, false, WS);
    bool result;
#if PRELOAD == 1
    printf("check memcpy result\n");
    vec_is_equal(&input[0][0][0][0], (elem_t*) A_copy_addr, sizeof(input)/sizeof(elem_t));
#endif

    if(sp_input_base_addr[thread_id] == -1){
      printf("check conv result\n");
      result = vec_is_equal(&output_mat[0][0], &output_mat_spad[0][0], sizeof(output_mat)/sizeof(elem_t));
    }
    else{
      printf("output to spad\n");
      
      for(int i = 0; i < N_PATCHES; i++)
        for(int j = 0; j < OUT_STRIDE; j++)
          output_mat_spad[i][j] = 0;
      
      uint64_t c_addr = sp_input_base_addr[thread_id] + BASE_ADDR;
      printf("c spad addr: 0x%08lx\n", c_addr);
      int out_len = BATCH_SIZE*OUT_ROW_DIM*OUT_COL_DIM*OUT_STRIDE;
      memcpy((elem_t*) output_mat_spad, (elem_t*) c_addr, sizeof(elem_t)*out_len);
      printf("check matmul result\n");
      result = vec_is_equal(&output_mat[0][0], (elem_t*) c_addr, sizeof(output_mat)/sizeof(elem_t));  
    }
    if (result == false){
        printf("gold: \n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");
        printf("C: \n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
                printf("%d,", output_mat_spad[orow][ocol]);
            }
            printf("\b],\n");
        }
    }
    exit(0);
}

