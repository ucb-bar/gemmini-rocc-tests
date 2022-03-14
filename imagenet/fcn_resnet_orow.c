#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "fcn_resnet_params.h"
#include "resnet50_mt_images.h"

#define num_proc 4
#define num_layer 56
#define num_resadd 16

#define THREAD_SYNC true // must do sync
#define BATCH_DIVIDE 1
#define OROW_DIVIDE 2 // 1: independent, 2: 2+2 collab, 4: sequential

#define A_no_max_block 0
#define B_no_max_block 0

#define priority true // ToDo: set it to true for priorized cores
#define target_util 70 // ToDo: needs to be changed for target utilization
#define bubble 0
#define target_util_resadd 25
#define target_util_fc 25


#define FCN_REPEAT 5


pthread_barrier_t barrier;

#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS true

//meaningless
static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t Out[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

struct thread_args{
    uint64_t total_thread_cycles, total_conv_cycles, total_matmul_cycles, total_resadd_cycles;
	uint64_t res_add_cycles[num_resadd];
	uint64_t conv_cycles[num_layer];
    uint64_t matmul_cycles[num_layer];
	uint64_t other_cycles; //global average
};
// random matmul to warm up thread
void *thread_matmul0(void *arg){
        struct thread_args * matmul_args = (struct thread_args *) arg;
        gemmini_flush(0);
        int cid = sched_getcpu();//matmul_args->i;
          elem_t* A = (elem_t*) in_A + MAT_DIM_K*(MAT_DIM_I/2)*(cid/2);
          elem_t* B = (elem_t*) in_B + (MAT_DIM_J/2)*(cid%2);
          elem_t* C = (elem_t*) Out + (MAT_DIM_J/2)*(cid%2) + MAT_DIM_J*(MAT_DIM_I/2)*(cid/2);
          tiled_matmul_auto(MAT_DIM_I/2, MAT_DIM_J/2, MAT_DIM_K,
                                A, B, NULL, C, //NO_BIAS ? NULL : D, C,
                           MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            false, false,
            false, !FULL_BIAS_WIDTH,
            WS);
}


void *thread_NN(void *arg){
	int cid = sched_getcpu();
	struct thread_args * nn_args = (struct thread_args *) arg;
    enum tiled_matmul_type_t tiled_matmul_type = WS;
	gemmini_flush(0);
    cid = cid % OROW_DIVIDE;
    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;
    //int image_offset = conv_1_params.in_channels * conv_1_params.in_dim * conv_1_params.in_dim * cid;
    pthread_barrier_wait(&barrier);
    
    uint64_t thread_start = read_cycles();
    
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_params.stride, conv_1_params.dilation, conv_1_params.padding, conv_1_params.kernel_size,
        conv_1_params.out_stride,

        (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

        RELU, conv_1_params.output_scale, 0,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params.I, conv_2_params.J, conv_2_params.K, conv_2_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,
        NO_ACTIVATION, conv_2_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_dim,
        conv_3_params.stride, conv_3_params.dilation, conv_3_params.padding, conv_3_params.kernel_size,
        conv_3_params.out_stride,

        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

        NO_ACTIVATION, conv_3_params.output_scale, 0,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_4
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_4_params.I, conv_4_params.J, conv_4_params.K, conv_4_params.out_stride,
        (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,
        RELU, conv_4_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Downsampling conv_1_out_pooled
    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params.I, conv_5_params.J, conv_5_params.K, conv_5_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,
        NO_ACTIVATION, conv_5_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_5_out,
        (elem_t*)conv_4_out,
        (elem_t*)conv_4_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_6
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_6_params.I, conv_6_params.J, conv_6_params.K, conv_6_params.out_stride,
        (elem_t*)conv_4_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,
        NO_ACTIVATION, conv_6_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_params.stride, conv_7_params.dilation, conv_7_params.padding, conv_7_params.kernel_size,
        conv_7_params.out_stride,

        (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

        NO_ACTIVATION, conv_7_params.output_scale, 0,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params.I, conv_8_params.J, conv_8_params.K, conv_8_params.out_stride,
        (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,
        RELU, conv_8_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_4_out,
        (elem_t*)conv_8_out,
        (elem_t*)conv_8_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params.I, conv_9_params.J, conv_9_params.K, conv_9_params.out_stride,
        (elem_t*)conv_8_out, (elem_t*)conv_9_w, (acc_t*)conv_9_b, (elem_t*)conv_9_out,
        NO_ACTIVATION, conv_9_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_10
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_10_params.batch_size, conv_10_params.in_dim, conv_10_params.in_channels,
        conv_10_params.out_channels, conv_10_params.out_dim,
        conv_10_params.stride, conv_10_params.dilation, conv_10_params.padding, conv_10_params.kernel_size,
        conv_10_params.out_stride,

        (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

        NO_ACTIVATION, conv_10_params.output_scale, 0,
        conv_10_params.pool_size, 0, conv_10_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params.I, conv_11_params.J, conv_11_params.K, conv_11_params.out_stride,
        (elem_t*)conv_10_out, (elem_t*)conv_11_w, (acc_t*)conv_11_b, (elem_t*)conv_11_out,
        RELU, conv_11_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_8_out,
        (elem_t*)conv_11_out,
        (elem_t*)conv_11_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_12
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_12_params.I, conv_12_params.J, conv_12_params.K, conv_12_params.out_stride,
        (elem_t*)conv_11_out, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_12_out,
        NO_ACTIVATION, conv_12_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_13
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
        conv_13_params.out_channels, conv_13_params.out_dim,
        conv_13_params.stride, conv_13_params.dilation, conv_13_params.padding, conv_13_params.kernel_size,
        conv_13_params.out_stride,

        (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

        NO_ACTIVATION, conv_13_params.output_scale, 0,
        conv_13_params.pool_size, 0, conv_13_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params.I, conv_14_params.J, conv_14_params.K, conv_14_params.out_stride,
        (elem_t*)conv_13_out, (elem_t*)conv_14_w, (acc_t*)conv_14_b, (elem_t*)conv_14_out,
        RELU, conv_14_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Downsampling conv_11_out
    // conv_15
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_15_params.batch_size, conv_15_params.in_dim, conv_15_params.in_channels,
        conv_15_params.out_channels, conv_15_params.out_dim,
        conv_15_params.stride, conv_15_params.dilation, conv_15_params.padding, conv_15_params.kernel_size,
        conv_15_params.out_stride,

        (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

        NO_ACTIVATION, conv_15_params.output_scale, 0,
        conv_15_params.pool_size, 0, conv_15_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_15_out,
        (elem_t*)conv_14_out,
        (elem_t*)conv_14_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_16
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_16_params.I, conv_16_params.J, conv_16_params.K, conv_16_params.out_stride,
        (elem_t*)conv_14_out, (elem_t*)conv_16_w, (acc_t*)conv_16_b, (elem_t*)conv_16_out,
        NO_ACTIVATION, conv_16_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_17_params.batch_size, conv_17_params.in_dim, conv_17_params.in_channels,
        conv_17_params.out_channels, conv_17_params.out_dim,
        conv_17_params.stride, conv_17_params.dilation, conv_17_params.padding, conv_17_params.kernel_size,
        conv_17_params.out_stride,

        (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

        NO_ACTIVATION, conv_17_params.output_scale, 0,
        conv_17_params.pool_size, 0, conv_17_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params.I, conv_18_params.J, conv_18_params.K, conv_18_params.out_stride,
        (elem_t*)conv_17_out, (elem_t*)conv_18_w, (acc_t*)conv_18_b, (elem_t*)conv_18_out,
        RELU, conv_18_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_14_out,
        (elem_t*)conv_18_out,
        (elem_t*)conv_18_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_19_params.I, conv_19_params.J, conv_19_params.K, conv_19_params.out_stride,
        (elem_t*)conv_18_out, (elem_t*)conv_19_w, (acc_t*)conv_19_b, (elem_t*)conv_19_out,
        NO_ACTIVATION, conv_19_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_20
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_20_params.batch_size, conv_20_params.in_dim, conv_20_params.in_channels,
        conv_20_params.out_channels, conv_20_params.out_dim,
        conv_20_params.stride, conv_20_params.dilation, conv_20_params.padding, conv_20_params.kernel_size,
        conv_20_params.out_stride,

        (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

        NO_ACTIVATION, conv_20_params.output_scale, 0,
        conv_20_params.pool_size, 0, conv_20_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params.I, conv_21_params.J, conv_21_params.K, conv_21_params.out_stride,
        (elem_t*)conv_20_out, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)conv_21_out,
        RELU, conv_21_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_18_out,
        (elem_t*)conv_21_out,
        (elem_t*)conv_21_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_22
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_22_params.I, conv_22_params.J, conv_22_params.K, conv_22_params.out_stride,
        (elem_t*)conv_21_out, (elem_t*)conv_22_w, (acc_t*)conv_22_b, (elem_t*)conv_22_out,
        NO_ACTIVATION, conv_22_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_23
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_23_params.batch_size, conv_23_params.in_dim, conv_23_params.in_channels,
        conv_23_params.out_channels, conv_23_params.out_dim,
        conv_23_params.stride, conv_23_params.dilation, conv_23_params.padding, conv_23_params.kernel_size,
        conv_23_params.out_stride,

        (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

        NO_ACTIVATION, conv_23_params.output_scale, 0,
        conv_23_params.pool_size, 0, conv_23_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params.I, conv_24_params.J, conv_24_params.K, conv_24_params.out_stride,
        (elem_t*)conv_23_out, (elem_t*)conv_24_w, (acc_t*)conv_24_b, (elem_t*)conv_24_out,
        RELU, conv_24_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_21_out,
        (elem_t*)conv_24_out,
        (elem_t*)conv_24_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_25
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_25_params.I, conv_25_params.J, conv_25_params.K, conv_25_params.out_stride,
        (elem_t*)conv_24_out, (elem_t*)conv_25_w, (acc_t*)conv_25_b, (elem_t*)conv_25_out,
        NO_ACTIVATION, conv_25_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_26
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_26_params.batch_size, conv_26_params.in_dim, conv_26_params.in_channels,
        conv_26_params.out_channels, conv_26_params.out_dim,
        conv_26_params.stride, conv_26_params.dilation, conv_26_params.padding, conv_26_params.kernel_size,
        conv_26_params.out_stride,

        (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

        NO_ACTIVATION, conv_26_params.output_scale, 0,
        conv_26_params.pool_size, 0, conv_26_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_27
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_27_params.I, conv_27_params.J, conv_27_params.K, conv_27_params.out_stride,
        (elem_t*)conv_26_out, (elem_t*)conv_27_w, (acc_t*)conv_27_b, (elem_t*)conv_27_out,
        RELU, conv_27_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Downsampling conv_24_out
    // conv_28
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_28_params.I, conv_28_params.J, conv_28_params.K, conv_28_params.out_stride,
        (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,
        NO_ACTIVATION, conv_28_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_28_out,
        (elem_t*)conv_27_out,
        (elem_t*)conv_27_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_29
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_29_params.I, conv_29_params.J, conv_29_params.K, conv_29_params.out_stride,
        (elem_t*)conv_27_out, (elem_t*)conv_29_w, (acc_t*)conv_29_b, (elem_t*)conv_29_out,
        NO_ACTIVATION, conv_29_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_30
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
        conv_30_params.out_channels, conv_30_params.out_dim,
        conv_30_params.stride, conv_30_params.dilation, conv_30_params.padding, conv_30_params.kernel_size,
        conv_30_params.out_stride,

        (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

        NO_ACTIVATION, conv_30_params.output_scale, 0,
        conv_30_params.pool_size, 0, conv_30_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_31
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_31_params.I, conv_31_params.J, conv_31_params.K, conv_31_params.out_stride,
        (elem_t*)conv_30_out, (elem_t*)conv_31_w, (acc_t*)conv_31_b, (elem_t*)conv_31_out,
        RELU, conv_31_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_27_out,
        (elem_t*)conv_31_out,
        (elem_t*)conv_31_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_32
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_32_params.I, conv_32_params.J, conv_32_params.K, conv_32_params.out_stride,
        (elem_t*)conv_31_out, (elem_t*)conv_32_w, (acc_t*)conv_32_b, (elem_t*)conv_32_out,
        NO_ACTIVATION, conv_32_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_33
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_33_params.batch_size, conv_33_params.in_dim, conv_33_params.in_channels,
        conv_33_params.out_channels, conv_33_params.out_dim,
        conv_33_params.stride, conv_33_params.dilation, conv_33_params.padding, conv_33_params.kernel_size,
        conv_33_params.out_stride,

        (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

        NO_ACTIVATION, conv_33_params.output_scale, 0,
        conv_33_params.pool_size, 0, conv_33_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_34
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_34_params.I, conv_34_params.J, conv_34_params.K, conv_34_params.out_stride,
        (elem_t*)conv_33_out, (elem_t*)conv_34_w, (acc_t*)conv_34_b, (elem_t*)conv_34_out,
        RELU, conv_34_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_31_out,
        (elem_t*)conv_34_out,
        (elem_t*)conv_34_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_35
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_35_params.I, conv_35_params.J, conv_35_params.K, conv_35_params.out_stride,
        (elem_t*)conv_34_out, (elem_t*)conv_35_w, (acc_t*)conv_35_b, (elem_t*)conv_35_out,
        NO_ACTIVATION, conv_35_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_36
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_36_params.batch_size, conv_36_params.in_dim, conv_36_params.in_channels,
        conv_36_params.out_channels, conv_36_params.out_dim,
        conv_36_params.stride, conv_36_params.dilation, conv_36_params.padding, conv_36_params.kernel_size,
        conv_36_params.out_stride,

        (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

        NO_ACTIVATION, conv_36_params.output_scale, 0,
        conv_36_params.pool_size, 0, conv_36_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_37
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_37_params.I, conv_37_params.J, conv_37_params.K, conv_37_params.out_stride,
        (elem_t*)conv_36_out, (elem_t*)conv_37_w, (acc_t*)conv_37_b, (elem_t*)conv_37_out,
        RELU, conv_37_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_34_out,
        (elem_t*)conv_37_out,
        (elem_t*)conv_37_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_38
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_38_params.I, conv_38_params.J, conv_38_params.K, conv_38_params.out_stride,
        (elem_t*)conv_37_out, (elem_t*)conv_38_w, (acc_t*)conv_38_b, (elem_t*)conv_38_out,
        NO_ACTIVATION, conv_38_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_39
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_39_params.batch_size, conv_39_params.in_dim, conv_39_params.in_channels,
        conv_39_params.out_channels, conv_39_params.out_dim,
        conv_39_params.stride, conv_39_params.dilation, conv_39_params.padding, conv_39_params.kernel_size,
        conv_39_params.out_stride,

        (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

        NO_ACTIVATION, conv_39_params.output_scale, 0,
        conv_39_params.pool_size, 0, conv_39_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_40
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_40_params.I, conv_40_params.J, conv_40_params.K, conv_40_params.out_stride,
        (elem_t*)conv_39_out, (elem_t*)conv_40_w, (acc_t*)conv_40_b, (elem_t*)conv_40_out,
        RELU, conv_40_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_37_out,
        (elem_t*)conv_40_out,
        (elem_t*)conv_40_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_41
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_41_params.I, conv_41_params.J, conv_41_params.K, conv_41_params.out_stride,
        (elem_t*)conv_40_out, (elem_t*)conv_41_w, (acc_t*)conv_41_b, (elem_t*)conv_41_out,
        NO_ACTIVATION, conv_41_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[26] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_42
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_42_params.batch_size, conv_42_params.in_dim, conv_42_params.in_channels,
        conv_42_params.out_channels, conv_42_params.out_dim,
        conv_42_params.stride, conv_42_params.dilation, conv_42_params.padding, conv_42_params.kernel_size,
        conv_42_params.out_stride,

        (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

        NO_ACTIVATION, conv_42_params.output_scale, 0,
        conv_42_params.pool_size, 0, conv_42_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_43
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_43_params.I, conv_43_params.J, conv_43_params.K, conv_43_params.out_stride,
        (elem_t*)conv_42_out, (elem_t*)conv_43_w, (acc_t*)conv_43_b, (elem_t*)conv_43_out,
        RELU, conv_43_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_40_out,
        (elem_t*)conv_43_out,
        (elem_t*)conv_43_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_44
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_44_params.I, conv_44_params.J, conv_44_params.K, conv_44_params.out_stride,
        (elem_t*)conv_43_out, (elem_t*)conv_44_w, (acc_t*)conv_44_b, (elem_t*)conv_44_out,
        NO_ACTIVATION, conv_44_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_45
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_45_params.batch_size, conv_45_params.in_dim, conv_45_params.in_channels,
        conv_45_params.out_channels, conv_45_params.out_dim,
        conv_45_params.stride, conv_45_params.dilation, conv_45_params.padding, conv_45_params.kernel_size,
        conv_45_params.out_stride,

        (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

        NO_ACTIVATION, conv_45_params.output_scale, 0,
        conv_45_params.pool_size, 0, conv_45_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_46
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_46_params.I, conv_46_params.J, conv_46_params.K, conv_46_params.out_stride,
        (elem_t*)conv_45_out, (elem_t*)conv_46_w, (acc_t*)conv_46_b, (elem_t*)conv_46_out,
        RELU, conv_46_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Downsampling conv_43_out
    // conv_47
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_47_params.I, conv_47_params.J, conv_47_params.K, conv_47_params.out_stride,
        (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,
        NO_ACTIVATION, conv_47_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[30] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_47_out,
        (elem_t*)conv_46_out,
        (elem_t*)conv_46_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_48
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_48_params.I, conv_48_params.J, conv_48_params.K, conv_48_params.out_stride,
        (elem_t*)conv_46_out, (elem_t*)conv_48_w, (acc_t*)conv_48_b, (elem_t*)conv_48_out,
        NO_ACTIVATION, conv_48_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[31] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_49
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_49_params.batch_size, conv_49_params.in_dim, conv_49_params.in_channels,
        conv_49_params.out_channels, conv_49_params.out_dim,
        conv_49_params.stride, conv_49_params.dilation, conv_49_params.padding, conv_49_params.kernel_size,
        conv_49_params.out_stride,

        (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

        NO_ACTIVATION, conv_49_params.output_scale, 0,
        conv_49_params.pool_size, 0, conv_49_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_50
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_50_params.I, conv_50_params.J, conv_50_params.K, conv_50_params.out_stride,
        (elem_t*)conv_49_out, (elem_t*)conv_50_w, (acc_t*)conv_50_b, (elem_t*)conv_50_out,
        RELU, conv_50_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[32] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_46_out,
        (elem_t*)conv_50_out,
        (elem_t*)conv_50_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_51
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_51_params.I, conv_51_params.J, conv_51_params.K, conv_51_params.out_stride,
        (elem_t*)conv_50_out, (elem_t*)conv_51_w, (acc_t*)conv_51_b, (elem_t*)conv_51_out,
        NO_ACTIVATION, conv_51_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[33] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_52
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_52_params.batch_size, conv_52_params.in_dim, conv_52_params.in_channels,
        conv_52_params.out_channels, conv_52_params.out_dim,
        conv_52_params.stride, conv_52_params.dilation, conv_52_params.padding, conv_52_params.kernel_size,
        conv_52_params.out_stride,

        (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

        NO_ACTIVATION, conv_52_params.output_scale, 0,
        conv_52_params.pool_size, 0, conv_52_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_53
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_53_params.I, conv_53_params.J, conv_53_params.K, conv_53_params.out_stride,
        (elem_t*)conv_52_out, (elem_t*)conv_53_w, (acc_t*)conv_53_b, (elem_t*)conv_53_out,
        RELU, conv_53_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[34] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_50_out,
        (elem_t*)conv_53_out,
        (elem_t*)conv_53_out,
        false,
         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_resadd, bubble);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_54
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_54_params.batch_size, conv_54_params.in_dim, conv_54_params.in_channels,
        conv_54_params.out_channels, conv_54_params.out_dim,
        conv_54_params.stride, conv_54_params.dilation, conv_54_params.padding, conv_54_params.kernel_size,
        conv_54_params.out_stride,

        (elem_t*)conv_53_out, (elem_t*)conv_54_w, (acc_t*)conv_54_b, (elem_t*)conv_54_out,

        RELU, conv_54_params.output_scale, 0,
        conv_54_params.pool_size, 0, conv_54_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_55
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_55_params.I, conv_55_params.J, conv_55_params.K, conv_55_params.out_stride,
        (elem_t*)conv_54_out, (elem_t*)conv_55_w, (acc_t*)conv_55_b, (elem_t*)conv_55_out,
        NO_ACTIVATION, conv_55_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[35] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif

// interpolation

    start = read_cycles();
    tiled_interpolate_auto(conv_55_params.in_dim, conv_55_params.out_stride, 224,
	(elem_t*) conv_55_out, (elem_t*) image_out, OROW_DIVIDE, cid);
    end = read_cycles();
    other_cycles = end - start;
    nn_args->other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
 
/*        
    // conv_56
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_56_params.batch_size, conv_56_params.in_dim, conv_56_params.in_channels,
        conv_56_params.out_channels, conv_56_params.out_dim,
        conv_56_params.stride, conv_56_params.dilation, conv_56_params.padding, conv_56_params.kernel_size,
        conv_56_params.out_stride,

        (elem_t*)conv_55_out, (elem_t*)conv_56_w, (acc_t*)conv_56_b, (elem_t*)conv_56_out,

        RELU, conv_56_params.output_scale, 0,
        conv_56_params.pool_size, 0, conv_56_params.pool_padding, false,

         WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
        
    // conv_57
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_57_params.I, conv_57_params.J, conv_57_params.K, conv_57_params.out_stride,
        (elem_t*)conv_56_out, (elem_t*)conv_57_w, (acc_t*)conv_57_b, (elem_t*)conv_57_out,
        NO_ACTIVATION, conv_57_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[36] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
  */      

    uint64_t thread_end = read_cycles();
    nn_args->total_thread_cycles = thread_end - thread_start;
    nn_args->total_matmul_cycles = matmul_cycles;
    nn_args->total_conv_cycles = conv_cycles;
    nn_args->other_cycles = other_cycles;
    nn_args->total_resadd_cycles = res_add_cycles;

}
void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
    printf("print msg - cpu_id: %d \n", cpu_id);
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    int cpu_id;
    cpu_id = sched_getcpu();
    cpu_set_t cpuset[num_proc];
    pthread_t thread[num_proc];
    pthread_attr_t attr[num_proc];
    for(int i = 0; i < num_proc; i++)
	pthread_attr_init(&attr[i]);
    struct thread_args nn_args[num_proc];


    for(int i = 0; i < num_proc; i++){
	 CPU_ZERO(&cpuset[i]);
	 CPU_SET(i, &cpuset[i]);
	 pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
	 pthread_create(&thread[i], &attr[i], print_message, NULL);
    }

    for(int i = 0; i < num_proc; i++){
        pthread_join(thread[i], NULL);
    }

    //just random turn
    for(int i = 0; i < num_proc; i++){
        pthread_create(&thread[i], &attr[i], thread_matmul0, &nn_args[i]);
    }

    for(int i = 0; i < num_proc; i++)
        pthread_join(thread[i], NULL);

    pthread_barrier_init(&barrier, NULL, OROW_DIVIDE);
        
    for(int r = 0; r < FCN_REPEAT; r++){
        uint64_t start = read_cycles();
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
        
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_join(thread[i], NULL);
        uint64_t end = read_cycles();

        uint64_t thread_resnet_max = 0;
        uint64_t total_resnet_max = 0;
        for(int i = 0; i < OROW_DIVIDE; i++){
            uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
            uint64_t conv_cycles = nn_args[i].total_conv_cycles;
            uint64_t res_add_cycles = nn_args[i].total_resadd_cycles;
            uint64_t other_cycles = nn_args[i].other_cycles;
            uint64_t total_cycles =  conv_cycles + matmul_cycles + res_add_cycles + other_cycles;
            uint64_t thread_cycles = nn_args[i].total_thread_cycles;

            thread_resnet_max = thread_resnet_max > thread_cycles ? thread_resnet_max : thread_cycles;
            total_resnet_max = total_resnet_max > total_cycles ? total_resnet_max : total_cycles;
        }
        printf("\nfcnresnet repeat %d total thread cycles: %llu\n", r, thread_resnet_max);
        printf("fcnresnet repeat %d total cycles: %llu\n", r, total_resnet_max);

        
        printf("worst case for each layers \n");
        

        for(int i = 0; i < 20; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].conv_cycles[i]) ? max : nn_args[j].conv_cycles[i];
            
            printf("fcnresnet repeat %d Conv layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        }
        

        for(int i = 0; i < 36; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].matmul_cycles[i]) ? max : nn_args[j].matmul_cycles[i];
            
            printf("fcnresnet repeat %d Matmul layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        

        }
        

        for(int i = 0; i < 16; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].res_add_cycles[i]) ? max : nn_args[j].res_add_cycles[i];
            
            printf("fcnresnet repeat %d Resadd layer %d worst cycles: %llu \n", r, i, max);
            max = 0;

            

        }
    
    }
    pthread_barrier_destroy(&barrier);
    printf("==================================\n");
    exit(0);
}

