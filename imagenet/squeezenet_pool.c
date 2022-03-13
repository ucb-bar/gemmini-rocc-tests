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

#include "squeezenet_orow_params.h"
#include "resnet50_mt_images.h"

#define num_proc 4
#define num_layer 27
#define num_resadd 1

#define THREAD_SYNC true // must do sync
#define BATCH_DIVIDE 1
#define OROW_DIVIDE 4 // 1: independent, 2: 2+2 collab, 4: sequential

#define A_no_max_block 0
#define B_no_max_block 0

#define priority false // ToDo: set it to true for priorized cores
#define target_util 0 // ToDo: needs to be changed for target utilization
#define bubble 0

#define SQUEEZENET_REPEAT 10

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
    uint64_t total_thread_cycles, total_conv_cycles, total_matmul_cycles, total_resadd_cycles, total_pool_cycles;
	uint64_t res_add_cycles[num_resadd];
	uint64_t conv_cycles[num_layer];
	uint64_t pool_cycles[num_layer];
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
	if(cid == 0 || cid == 1)
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
        conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
        conv_1_params.out_stride,

        (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

        RELU, conv_1_params.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, false,

        WS, 2* OROW_DIVIDE, BATCH_DIVIDE,  cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[0] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(&barrier);
#endif        
 // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
        conv_1_params.out_stride,

        (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

        RELU, conv_1_params.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, false,

        WS, OROW_DIVIDE * 2, BATCH_DIVIDE,  OROW_DIVIDE + cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[11] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(&barrier);
#endif   
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params.batch_size,
        conv_1_params.out_channels, conv_1_params.out_dim, conv_1_params.out_dim_pooled,
        conv_1_params.out_stride,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

        (elem_t*)conv_1_out, (elem_t*)conv_1_out_pooled,
	OROW_DIVIDE, BATCH_DIVIDE, cid);

    end = read_cycles();
    pool_cycles += end - start;
    nn_args->pool_cycles[0] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(&barrier);
#endif         
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params.I, conv_2_params.J, conv_2_params.K, conv_2_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,
        RELU, conv_2_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);
pthread_barrier_wait(&barrier);


    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_3_params.I, conv_3_params.J, conv_3_params.K, conv_3_params.out_stride,
        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_4_out,
        RELU, conv_3_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_4
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_4_params.batch_size, conv_4_params.in_dim, conv_4_params.in_channels,
        conv_4_params.out_channels, conv_4_params.out_dim,
        conv_4_params.stride, 1, conv_4_params.padding, conv_4_params.kernel_size, conv_4_params.out_stride,

        (elem_t*)conv_2_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out + conv_4_params.out_channels,

        RELU, conv_4_params.output_scale, 0,
        conv_4_params.pool_size, 0, conv_4_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params.I, conv_5_params.J, conv_5_params.K, conv_5_params.out_stride,
        (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,
        RELU, conv_5_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_6
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_6_params.batch_size, conv_6_params.in_dim, conv_6_params.in_channels,
        conv_6_params.out_channels, conv_6_params.out_dim,
        conv_6_params.stride, 1, conv_6_params.padding, conv_6_params.kernel_size, conv_6_params.out_stride,

        (elem_t*)conv_5_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_7_out_pooled,

        RELU, conv_6_params.output_scale, 0,
        conv_6_params.pool_size, conv_6_params.pool_stride, conv_6_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_7
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_params.stride, 1, conv_7_params.padding, conv_7_params.kernel_size, conv_7_params.out_stride,

        (elem_t*)conv_5_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out_pooled + conv_7_params.out_channels,

        RELU, conv_7_params.output_scale, 0,
        conv_7_params.pool_size, conv_7_params.pool_stride, conv_7_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params.I, conv_8_params.J, conv_8_params.K, conv_8_params.out_stride,
        (elem_t*)conv_7_out_pooled, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,
        RELU, conv_8_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params.I, conv_9_params.J, conv_9_params.K, conv_9_params.out_stride,
        (elem_t*)conv_8_out, (elem_t*)conv_9_w, (acc_t*)conv_9_b, (elem_t*)conv_10_out,
        RELU, conv_9_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_10
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_10_params.batch_size, conv_10_params.in_dim, conv_10_params.in_channels,
        conv_10_params.out_channels, conv_10_params.out_dim,
        conv_10_params.stride, 1, conv_10_params.padding, conv_10_params.kernel_size, conv_10_params.out_stride,

        (elem_t*)conv_8_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out + conv_10_params.out_channels,

        RELU, conv_10_params.output_scale, 0,
        conv_10_params.pool_size, 0, conv_10_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[4] = end - start;
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
    nn_args->matmul_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_12
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_12_params.batch_size, conv_12_params.in_dim, conv_12_params.in_channels,
        conv_12_params.out_channels, conv_12_params.out_dim,
        conv_12_params.stride, 1, conv_12_params.padding, conv_12_params.kernel_size, conv_12_params.out_stride,

        (elem_t*)conv_11_out, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_13_out_pooled,

        RELU, conv_12_params.output_scale, 0,
        conv_12_params.pool_size, conv_12_params.pool_stride, conv_12_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_13
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
        conv_13_params.out_channels, conv_13_params.out_dim,
        conv_13_params.stride, 1, conv_13_params.padding, conv_13_params.kernel_size, conv_13_params.out_stride,

        (elem_t*)conv_11_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out_pooled + conv_13_params.out_channels,

        RELU, conv_13_params.output_scale, 0,
        conv_13_params.pool_size, conv_13_params.pool_stride, conv_13_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params.I, conv_14_params.J, conv_14_params.K, conv_14_params.out_stride,
        (elem_t*)conv_13_out_pooled, (elem_t*)conv_14_w, (acc_t*)conv_14_b, (elem_t*)conv_14_out,
        RELU, conv_14_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_15_params.I, conv_15_params.J, conv_15_params.K, conv_15_params.out_stride,
        (elem_t*)conv_14_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_16_out,
        RELU, conv_15_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_16
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_16_params.batch_size, 13, 48,
        192, 13,
        1, 1, 1, 3, conv_16_params.out_stride,

        (elem_t*)conv_14_out, (elem_t*)conv_16_w, (acc_t*)conv_16_b, (elem_t*)conv_16_out + 192,

        RELU, conv_16_params.output_scale, 0,
        conv_16_params.pool_size, 0, conv_16_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


pthread_barrier_wait(&barrier);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_17
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_17_params.I, conv_17_params.J, conv_17_params.K, conv_17_params.out_stride,
        (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,
        RELU, conv_17_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params.I, conv_18_params.J, conv_18_params.K, conv_18_params.out_stride,
        (elem_t*)conv_17_out, (elem_t*)conv_18_w, (acc_t*)conv_18_b, (elem_t*)conv_19_out,
        RELU, conv_18_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_19
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_19_params.batch_size, conv_19_params.in_dim, conv_19_params.in_channels,
        conv_19_params.out_channels, conv_19_params.out_dim,
        conv_19_params.stride, 1, conv_19_params.padding, conv_19_params.kernel_size, conv_19_params.out_stride,

        (elem_t*)conv_17_out, (elem_t*)conv_19_w, (acc_t*)conv_19_b, (elem_t*)conv_19_out + conv_19_params.out_channels,

        RELU, conv_19_params.output_scale, 0,
        conv_19_params.pool_size, 0, conv_19_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


pthread_barrier_wait(&barrier);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_20
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_20_params.I, conv_20_params.J, conv_20_params.K, conv_20_params.out_stride,
        (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,
        RELU, conv_20_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params.I, conv_21_params.J, conv_21_params.K, conv_21_params.out_stride,
        (elem_t*)conv_20_out, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)conv_22_out,
        RELU, conv_21_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_22
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_22_params.batch_size, conv_22_params.in_dim, conv_22_params.in_channels,
        conv_22_params.out_channels, conv_22_params.out_dim,
        conv_22_params.stride, 1, conv_22_params.padding, conv_22_params.kernel_size, conv_22_params.out_stride,

        (elem_t*)conv_20_out, (elem_t*)conv_22_w, (acc_t*)conv_22_b, (elem_t*)conv_22_out + conv_22_params.out_channels,

        RELU, conv_22_params.output_scale, 0,
        conv_22_params.pool_size, 0, conv_22_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_23
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_23_params.I, conv_23_params.J, conv_23_params.K, conv_23_params.out_stride,
        (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,
        RELU, conv_23_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params.I, conv_24_params.J, conv_24_params.K, conv_24_params.out_stride,
        (elem_t*)conv_23_out, (elem_t*)conv_24_w, (acc_t*)conv_24_b, (elem_t*)conv_25_out,
        RELU, conv_24_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        


    // conv_25
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_25_params.batch_size, conv_25_params.in_dim, conv_25_params.in_channels,
        conv_25_params.out_channels, conv_25_params.out_dim,
        conv_25_params.stride, 1, conv_25_params.padding, conv_25_params.kernel_size, conv_25_params.out_stride,

        (elem_t*)conv_23_out, (elem_t*)conv_25_w, (acc_t*)conv_25_b, (elem_t*)conv_25_out + conv_25_params.out_channels,

        RELU, conv_25_params.output_scale, 0,
        conv_25_params.pool_size, 0, conv_25_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);


    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_26
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_26_params.I, conv_26_params.J, conv_26_params.K, conv_26_params.out_stride,
        (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,
        RELU, conv_26_params.output_scale, 0, true,
        tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif
        

    uint64_t thread_end = read_cycles();
    nn_args->total_thread_cycles = thread_end - thread_start;
    nn_args->total_matmul_cycles = matmul_cycles;
    nn_args->total_pool_cycles = pool_cycles;
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

    bool conv = true;
    bool check = false;
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
    
    for(int r = 0; r < SQUEEZENET_REPEAT; r++){
        uint64_t start = read_cycles();
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
        
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_join(thread[i], NULL);
        uint64_t end = read_cycles();

        printf("squeezenet repeat %d total cycles with threading overhead: %llu \n", r, end - start);

        
        uint64_t thread_resnet_max = 0;
        uint64_t total_resnet_max = 0;
        for(int i = 0; i < OROW_DIVIDE; i++){
            uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
            uint64_t conv_cycles = nn_args[i].total_conv_cycles;
	    uint64_t pool_cycles = nn_args[i].total_pool_cycles;
            uint64_t total_cycles =  conv_cycles + matmul_cycles + pool_cycles;
            uint64_t thread_cycles = nn_args[i].total_thread_cycles;
            thread_resnet_max = thread_resnet_max > thread_cycles ? thread_resnet_max : thread_cycles;
            total_resnet_max = total_resnet_max > total_cycles ? total_resnet_max : total_cycles;
        }
        printf("\nsqueezenet repeat %d total thread cycles: %llu\n", r, thread_resnet_max);
        printf("squeezenet repeat %d total cycles: %llu\n", r, total_resnet_max);
                                                    
        printf("worst case for each layers \n");
        

        for(int i = 0; i < 12; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].conv_cycles[i]) ? max : nn_args[j].conv_cycles[i];
            
            printf("squeezenet repeat %d Conv layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        }
        

        for(int i = 0; i < 15; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].matmul_cycles[i]) ? max : nn_args[j].matmul_cycles[i];
            
            printf("squeezenet repeat %d Matmul layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        

        }
        for(int i = 0; i < 1; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].pool_cycles[i]) ? max : nn_args[j].pool_cycles[i];
            
            printf("squeezenet repeat %d Pool layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        

        }
 
    
    }
    printf("==================================\n");
    pthread_barrier_destroy(&barrier);
    exit(0);
}

