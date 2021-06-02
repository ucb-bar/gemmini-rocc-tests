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

#include "densenet121_orow_params.h"
#include "resnet50_mt_images.h"

#define num_proc 4
#define num_layer 121 
#define num_resadd 0

#define THREAD_SYNC true // must do sync
#define BATCH_DIVIDE 1
#define OROW_DIVIDE 4 // 1: independent, 2: 2+2 collab, 4: sequential

#define SKIP_A false
#define SKIP_B false
#define SKIP_WEIGHT false //later

#define SKIP_A_DEN121 false
#define SKIP_B_DEN121 false
#define SKIP_IMAGE_DEN121 SKIP_A_DEN121
#define SKIP_WEIGHT_DEN121 SKIP_B_DEN121

#define PROFILE_DEN121 false
#define LATENCY_DEN121 1600
#define ALERT_CYCLE_DEN121 24
#define UNLOCK_CYCLE_DEN121 4
#define PAUSE_TURN_DEN121 2


pthread_barrier_t barrier;
pthread_mutex_t dense121_mutex;
int dense121_mutex_counter = 0;

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

    tiled_conv_A_stride_auto_bubble(
        conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
        256,

        (elem_t*)images, (elem_t*)conv_1_w, ((acc_t*)conv_1_b + 0), ((elem_t*)conv_1_out_pooled + 0),

        RELU, conv_1_params.output_scale, 0,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
	//    pthread_barrier_wait(&barrier);
	pthread_mutex_lock(&dense121_mutex);
	dense121_mutex_counter += 1;
	printf("dense121 mutex: %d, while loop cid %d\n", dense121_mutex_counter, cid);
	pthread_mutex_unlock(&dense121_mutex);

	while(dense121_mutex_counter%OROW_DIVIDE != 0){
		//printf("dense121 mutex: %d, while loop cid %d\n", dense121_mutex_counter, cid);
	}
	printf("mutex finished\n");
#endif  


    // dense_block 1
    // dense_layer 1
    // conv_2
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_2_params.I, conv_2_params.J, conv_2_params.K, 
            conv_2_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
	//        pthread_barrier_wait(&barrier);
	 
	pthread_mutex_lock(&dense121_mutex);

	dense121_mutex_counter += 1;
	pthread_mutex_unlock(&dense121_mutex);
	while(dense121_mutex_counter%OROW_DIVIDE != 0){
	}   
#endif
        
    // conv_3
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_dim,
        conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,
        256,

        (elem_t*)conv_2_out, (elem_t*)conv_3_w, ((acc_t*)conv_1_b + 64), ((elem_t*)conv_1_out_pooled + 64),

        RELU, conv_3_params.output_scale, 0,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 2
    // conv_4
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_4_params.I, conv_4_params.J, conv_4_params.K, 
            conv_4_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,
            RELU, conv_4_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[1] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_5
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_5_params.batch_size, conv_5_params.in_dim, conv_5_params.in_channels,
        conv_5_params.out_channels, conv_5_params.out_dim,
        conv_5_params.stride, 1, conv_5_params.padding, conv_5_params.kernel_size,
        256,

        (elem_t*)conv_4_out, (elem_t*)conv_5_w, ((acc_t*)conv_1_b + 96), ((elem_t*)conv_1_out_pooled + 96),

        RELU, conv_5_params.output_scale, 0,
        conv_5_params.pool_size, 0, conv_5_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 3
    // conv_6
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_6_params.I, conv_6_params.J, conv_6_params.K, 
            conv_6_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[2] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_7
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_params.stride, 1, conv_7_params.padding, conv_7_params.kernel_size,
        256,

        (elem_t*)conv_6_out, (elem_t*)conv_7_w, ((acc_t*)conv_1_b + 128), ((elem_t*)conv_1_out_pooled + 128),

        RELU, conv_7_params.output_scale, 0,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 4
    // conv_8
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_8_params.I, conv_8_params.J, conv_8_params.K, 
            conv_8_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,
            RELU, conv_8_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[3] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_9
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_9_params.batch_size, conv_9_params.in_dim, conv_9_params.in_channels,
        conv_9_params.out_channels, conv_9_params.out_dim,
        conv_9_params.stride, 1, conv_9_params.padding, conv_9_params.kernel_size,
        256,

        (elem_t*)conv_8_out, (elem_t*)conv_9_w, ((acc_t*)conv_1_b + 160), ((elem_t*)conv_1_out_pooled + 160),

        RELU, conv_9_params.output_scale, 0,
        conv_9_params.pool_size, 0, conv_9_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 5
    // conv_10
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_10_params.I, conv_10_params.J, conv_10_params.K, 
            conv_10_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,
            RELU, conv_10_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[4] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_11
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_11_params.batch_size, conv_11_params.in_dim, conv_11_params.in_channels,
        conv_11_params.out_channels, conv_11_params.out_dim,
        conv_11_params.stride, 1, conv_11_params.padding, conv_11_params.kernel_size,
        256,

        (elem_t*)conv_10_out, (elem_t*)conv_11_w, ((acc_t*)conv_1_b + 192), ((elem_t*)conv_1_out_pooled + 192),

        RELU, conv_11_params.output_scale, 0,
        conv_11_params.pool_size, 0, conv_11_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 6
    // conv_12
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_12_params.I, conv_12_params.J, conv_12_params.K, 
            conv_12_params.J, 256,
            (elem_t*)conv_1_out_pooled, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[5] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_13
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
        conv_13_params.out_channels, conv_13_params.out_dim,
        conv_13_params.stride, 1, conv_13_params.padding, conv_13_params.kernel_size,
        256,

        (elem_t*)conv_12_out, (elem_t*)conv_13_w, ((acc_t*)conv_1_b + 224), ((elem_t*)conv_1_out_pooled + 224),

        RELU, conv_13_params.output_scale, 0,
        conv_13_params.pool_size, 0, conv_13_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_14
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_14_params.batch_size, conv_14_params.in_dim, conv_14_params.in_channels,
        conv_14_params.out_channels, conv_14_params.out_dim,
        conv_14_params.stride, 1, conv_14_params.padding, conv_14_params.kernel_size,
        512,

        (elem_t*)conv_1_out_pooled, (elem_t*)conv_14_w, ((acc_t*)conv_14_b + 0), ((elem_t*)conv_14_out_pooled + 0),

        RELU, conv_14_params.output_scale, 0,
        conv_14_params.pool_size, conv_14_params.pool_stride, conv_14_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_block 2
    // dense_layer 1
    // conv_15
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_15_params.I, conv_15_params.J, conv_15_params.K, 
            conv_15_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,
            RELU, conv_15_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[6] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_16
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_16_params.batch_size, conv_16_params.in_dim, conv_16_params.in_channels,
        conv_16_params.out_channels, conv_16_params.out_dim,
        conv_16_params.stride, 1, conv_16_params.padding, conv_16_params.kernel_size,
        512,

        (elem_t*)conv_15_out, (elem_t*)conv_16_w, ((acc_t*)conv_14_b + 128), ((elem_t*)conv_14_out_pooled + 128),

        RELU, conv_16_params.output_scale, 0,
        conv_16_params.pool_size, 0, conv_16_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 2
    // conv_17
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_17_params.I, conv_17_params.J, conv_17_params.K, 
            conv_17_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,
            RELU, conv_17_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[7] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_18
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_18_params.batch_size, conv_18_params.in_dim, conv_18_params.in_channels,
        conv_18_params.out_channels, conv_18_params.out_dim,
        conv_18_params.stride, 1, conv_18_params.padding, conv_18_params.kernel_size,
        512,

        (elem_t*)conv_17_out, (elem_t*)conv_18_w, ((acc_t*)conv_14_b + 160), ((elem_t*)conv_14_out_pooled + 160),

        RELU, conv_18_params.output_scale, 0,
        conv_18_params.pool_size, 0, conv_18_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 3
    // conv_19
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_19_params.I, conv_19_params.J, conv_19_params.K, 
            conv_19_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_19_w, (acc_t*)conv_19_b, (elem_t*)conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[8] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_20
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_20_params.batch_size, conv_20_params.in_dim, conv_20_params.in_channels,
        conv_20_params.out_channels, conv_20_params.out_dim,
        conv_20_params.stride, 1, conv_20_params.padding, conv_20_params.kernel_size,
        512,

        (elem_t*)conv_19_out, (elem_t*)conv_20_w, ((acc_t*)conv_14_b + 192), ((elem_t*)conv_14_out_pooled + 192),

        RELU, conv_20_params.output_scale, 0,
        conv_20_params.pool_size, 0, conv_20_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 4
    // conv_21
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_21_params.I, conv_21_params.J, conv_21_params.K, 
            conv_21_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)conv_21_out,
            RELU, conv_21_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[9] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_22
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_22_params.batch_size, conv_22_params.in_dim, conv_22_params.in_channels,
        conv_22_params.out_channels, conv_22_params.out_dim,
        conv_22_params.stride, 1, conv_22_params.padding, conv_22_params.kernel_size,
        512,

        (elem_t*)conv_21_out, (elem_t*)conv_22_w, ((acc_t*)conv_14_b + 224), ((elem_t*)conv_14_out_pooled + 224),

        RELU, conv_22_params.output_scale, 0,
        conv_22_params.pool_size, 0, conv_22_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 5
    // conv_23
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_23_params.I, conv_23_params.J, conv_23_params.K, 
            conv_23_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,
            RELU, conv_23_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[10] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_24
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_24_params.batch_size, conv_24_params.in_dim, conv_24_params.in_channels,
        conv_24_params.out_channels, conv_24_params.out_dim,
        conv_24_params.stride, 1, conv_24_params.padding, conv_24_params.kernel_size,
        512,

        (elem_t*)conv_23_out, (elem_t*)conv_24_w, ((acc_t*)conv_14_b + 256), ((elem_t*)conv_14_out_pooled + 256),

        RELU, conv_24_params.output_scale, 0,
        conv_24_params.pool_size, 0, conv_24_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 6
    // conv_25
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_25_params.I, conv_25_params.J, conv_25_params.K, 
            conv_25_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_25_w, (acc_t*)conv_25_b, (elem_t*)conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[11] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_26
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_26_params.batch_size, conv_26_params.in_dim, conv_26_params.in_channels,
        conv_26_params.out_channels, conv_26_params.out_dim,
        conv_26_params.stride, 1, conv_26_params.padding, conv_26_params.kernel_size,
        512,

        (elem_t*)conv_25_out, (elem_t*)conv_26_w, ((acc_t*)conv_14_b + 288), ((elem_t*)conv_14_out_pooled + 288),

        RELU, conv_26_params.output_scale, 0,
        conv_26_params.pool_size, 0, conv_26_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 7
    // conv_27
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_27_params.I, conv_27_params.J, conv_27_params.K, 
            conv_27_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_27_w, (acc_t*)conv_27_b, (elem_t*)conv_27_out,
            RELU, conv_27_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[12] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_28
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_28_params.batch_size, conv_28_params.in_dim, conv_28_params.in_channels,
        conv_28_params.out_channels, conv_28_params.out_dim,
        conv_28_params.stride, 1, conv_28_params.padding, conv_28_params.kernel_size,
        512,

        (elem_t*)conv_27_out, (elem_t*)conv_28_w, ((acc_t*)conv_14_b + 320), ((elem_t*)conv_14_out_pooled + 320),

        RELU, conv_28_params.output_scale, 0,
        conv_28_params.pool_size, 0, conv_28_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 8
    // conv_29
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_29_params.I, conv_29_params.J, conv_29_params.K, 
            conv_29_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_29_w, (acc_t*)conv_29_b, (elem_t*)conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[13] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_30
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
        conv_30_params.out_channels, conv_30_params.out_dim,
        conv_30_params.stride, 1, conv_30_params.padding, conv_30_params.kernel_size,
        512,

        (elem_t*)conv_29_out, (elem_t*)conv_30_w, ((acc_t*)conv_14_b + 352), ((elem_t*)conv_14_out_pooled + 352),

        RELU, conv_30_params.output_scale, 0,
        conv_30_params.pool_size, 0, conv_30_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 9
    // conv_31
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_31_params.I, conv_31_params.J, conv_31_params.K, 
            conv_31_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_31_w, (acc_t*)conv_31_b, (elem_t*)conv_31_out,
            RELU, conv_31_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[14] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_32
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_32_params.batch_size, conv_32_params.in_dim, conv_32_params.in_channels,
        conv_32_params.out_channels, conv_32_params.out_dim,
        conv_32_params.stride, 1, conv_32_params.padding, conv_32_params.kernel_size,
        512,

        (elem_t*)conv_31_out, (elem_t*)conv_32_w, ((acc_t*)conv_14_b + 384), ((elem_t*)conv_14_out_pooled + 384),

        RELU, conv_32_params.output_scale, 0,
        conv_32_params.pool_size, 0, conv_32_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 10
    // conv_33
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_33_params.I, conv_33_params.J, conv_33_params.K, 
            conv_33_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,
            RELU, conv_33_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[15] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_34
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_34_params.batch_size, conv_34_params.in_dim, conv_34_params.in_channels,
        conv_34_params.out_channels, conv_34_params.out_dim,
        conv_34_params.stride, 1, conv_34_params.padding, conv_34_params.kernel_size,
        512,

        (elem_t*)conv_33_out, (elem_t*)conv_34_w, ((acc_t*)conv_14_b + 416), ((elem_t*)conv_14_out_pooled + 416),

        RELU, conv_34_params.output_scale, 0,
        conv_34_params.pool_size, 0, conv_34_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 11
    // conv_35
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_35_params.I, conv_35_params.J, conv_35_params.K, 
            conv_35_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_35_w, (acc_t*)conv_35_b, (elem_t*)conv_35_out,
            RELU, conv_35_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[16] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_36
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_36_params.batch_size, conv_36_params.in_dim, conv_36_params.in_channels,
        conv_36_params.out_channels, conv_36_params.out_dim,
        conv_36_params.stride, 1, conv_36_params.padding, conv_36_params.kernel_size,
        512,

        (elem_t*)conv_35_out, (elem_t*)conv_36_w, ((acc_t*)conv_14_b + 448), ((elem_t*)conv_14_out_pooled + 448),

        RELU, conv_36_params.output_scale, 0,
        conv_36_params.pool_size, 0, conv_36_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 12
    // conv_37
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_37_params.I, conv_37_params.J, conv_37_params.K, 
            conv_37_params.J, 512,
            (elem_t*)conv_14_out_pooled, (elem_t*)conv_37_w, (acc_t*)conv_37_b, (elem_t*)conv_37_out,
            RELU, conv_37_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[17] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_38
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_38_params.batch_size, conv_38_params.in_dim, conv_38_params.in_channels,
        conv_38_params.out_channels, conv_38_params.out_dim,
        conv_38_params.stride, 1, conv_38_params.padding, conv_38_params.kernel_size,
        512,

        (elem_t*)conv_37_out, (elem_t*)conv_38_w, ((acc_t*)conv_14_b + 480), ((elem_t*)conv_14_out_pooled + 480),

        RELU, conv_38_params.output_scale, 0,
        conv_38_params.pool_size, 0, conv_38_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_39
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_39_params.batch_size, conv_39_params.in_dim, conv_39_params.in_channels,
        conv_39_params.out_channels, conv_39_params.out_dim,
        conv_39_params.stride, 1, conv_39_params.padding, conv_39_params.kernel_size,
        1024,

        (elem_t*)conv_14_out_pooled, (elem_t*)conv_39_w, ((acc_t*)conv_39_b + 0), ((elem_t*)conv_39_out_pooled + 0),

        RELU, conv_39_params.output_scale, 0,
        conv_39_params.pool_size, conv_39_params.pool_stride, conv_39_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_block 3
    // dense_layer 1
    // conv_40
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_40_params.I, conv_40_params.J, conv_40_params.K, 
            conv_40_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_40_w, (acc_t*)conv_40_b, (elem_t*)conv_40_out,
            RELU, conv_40_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[18] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_41
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_41_params.batch_size, conv_41_params.in_dim, conv_41_params.in_channels,
        conv_41_params.out_channels, conv_41_params.out_dim,
        conv_41_params.stride, 1, conv_41_params.padding, conv_41_params.kernel_size,
        1024,

        (elem_t*)conv_40_out, (elem_t*)conv_41_w, ((acc_t*)conv_39_b + 256), ((elem_t*)conv_39_out_pooled + 256),

        RELU, conv_41_params.output_scale, 0,
        conv_41_params.pool_size, 0, conv_41_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 2
    // conv_42
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_42_params.I, conv_42_params.J, conv_42_params.K, 
            conv_42_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,
            RELU, conv_42_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[19] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_43
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_43_params.batch_size, conv_43_params.in_dim, conv_43_params.in_channels,
        conv_43_params.out_channels, conv_43_params.out_dim,
        conv_43_params.stride, 1, conv_43_params.padding, conv_43_params.kernel_size,
        1024,

        (elem_t*)conv_42_out, (elem_t*)conv_43_w, ((acc_t*)conv_39_b + 288), ((elem_t*)conv_39_out_pooled + 288),

        RELU, conv_43_params.output_scale, 0,
        conv_43_params.pool_size, 0, conv_43_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 3
    // conv_44
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_44_params.I, conv_44_params.J, conv_44_params.K, 
            conv_44_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_44_w, (acc_t*)conv_44_b, (elem_t*)conv_44_out,
            RELU, conv_44_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[20] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_45
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_45_params.batch_size, conv_45_params.in_dim, conv_45_params.in_channels,
        conv_45_params.out_channels, conv_45_params.out_dim,
        conv_45_params.stride, 1, conv_45_params.padding, conv_45_params.kernel_size,
        1024,

        (elem_t*)conv_44_out, (elem_t*)conv_45_w, ((acc_t*)conv_39_b + 320), ((elem_t*)conv_39_out_pooled + 320),

        RELU, conv_45_params.output_scale, 0,
        conv_45_params.pool_size, 0, conv_45_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 4
    // conv_46
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_46_params.I, conv_46_params.J, conv_46_params.K, 
            conv_46_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_46_w, (acc_t*)conv_46_b, (elem_t*)conv_46_out,
            RELU, conv_46_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[21] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_47
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_47_params.batch_size, conv_47_params.in_dim, conv_47_params.in_channels,
        conv_47_params.out_channels, conv_47_params.out_dim,
        conv_47_params.stride, 1, conv_47_params.padding, conv_47_params.kernel_size,
        1024,

        (elem_t*)conv_46_out, (elem_t*)conv_47_w, ((acc_t*)conv_39_b + 352), ((elem_t*)conv_39_out_pooled + 352),

        RELU, conv_47_params.output_scale, 0,
        conv_47_params.pool_size, 0, conv_47_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 5
    // conv_48
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_48_params.I, conv_48_params.J, conv_48_params.K, 
            conv_48_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_48_w, (acc_t*)conv_48_b, (elem_t*)conv_48_out,
            RELU, conv_48_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[22] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_49
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_49_params.batch_size, conv_49_params.in_dim, conv_49_params.in_channels,
        conv_49_params.out_channels, conv_49_params.out_dim,
        conv_49_params.stride, 1, conv_49_params.padding, conv_49_params.kernel_size,
        1024,

        (elem_t*)conv_48_out, (elem_t*)conv_49_w, ((acc_t*)conv_39_b + 384), ((elem_t*)conv_39_out_pooled + 384),

        RELU, conv_49_params.output_scale, 0,
        conv_49_params.pool_size, 0, conv_49_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 6
    // conv_50
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_50_params.I, conv_50_params.J, conv_50_params.K, 
            conv_50_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_50_w, (acc_t*)conv_50_b, (elem_t*)conv_50_out,
            RELU, conv_50_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[23] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_51
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_51_params.batch_size, conv_51_params.in_dim, conv_51_params.in_channels,
        conv_51_params.out_channels, conv_51_params.out_dim,
        conv_51_params.stride, 1, conv_51_params.padding, conv_51_params.kernel_size,
        1024,

        (elem_t*)conv_50_out, (elem_t*)conv_51_w, ((acc_t*)conv_39_b + 416), ((elem_t*)conv_39_out_pooled + 416),

        RELU, conv_51_params.output_scale, 0,
        conv_51_params.pool_size, 0, conv_51_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 7
    // conv_52
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_52_params.I, conv_52_params.J, conv_52_params.K, 
            conv_52_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,
            RELU, conv_52_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[24] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_53
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_53_params.batch_size, conv_53_params.in_dim, conv_53_params.in_channels,
        conv_53_params.out_channels, conv_53_params.out_dim,
        conv_53_params.stride, 1, conv_53_params.padding, conv_53_params.kernel_size,
        1024,

        (elem_t*)conv_52_out, (elem_t*)conv_53_w, ((acc_t*)conv_39_b + 448), ((elem_t*)conv_39_out_pooled + 448),

        RELU, conv_53_params.output_scale, 0,
        conv_53_params.pool_size, 0, conv_53_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 8
    // conv_54
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_54_params.I, conv_54_params.J, conv_54_params.K, 
            conv_54_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_54_w, (acc_t*)conv_54_b, (elem_t*)conv_54_out,
            RELU, conv_54_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[25] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_55
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_55_params.batch_size, conv_55_params.in_dim, conv_55_params.in_channels,
        conv_55_params.out_channels, conv_55_params.out_dim,
        conv_55_params.stride, 1, conv_55_params.padding, conv_55_params.kernel_size,
        1024,

        (elem_t*)conv_54_out, (elem_t*)conv_55_w, ((acc_t*)conv_39_b + 480), ((elem_t*)conv_39_out_pooled + 480),

        RELU, conv_55_params.output_scale, 0,
        conv_55_params.pool_size, 0, conv_55_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 9
    // conv_56
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_56_params.I, conv_56_params.J, conv_56_params.K, 
            conv_56_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_56_w, (acc_t*)conv_56_b, (elem_t*)conv_56_out,
            RELU, conv_56_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[26] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_57
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_57_params.batch_size, conv_57_params.in_dim, conv_57_params.in_channels,
        conv_57_params.out_channels, conv_57_params.out_dim,
        conv_57_params.stride, 1, conv_57_params.padding, conv_57_params.kernel_size,
        1024,

        (elem_t*)conv_56_out, (elem_t*)conv_57_w, ((acc_t*)conv_39_b + 512), ((elem_t*)conv_39_out_pooled + 512),

        RELU, conv_57_params.output_scale, 0,
        conv_57_params.pool_size, 0, conv_57_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 10
    // conv_58
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_58_params.I, conv_58_params.J, conv_58_params.K, 
            conv_58_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_58_w, (acc_t*)conv_58_b, (elem_t*)conv_58_out,
            RELU, conv_58_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[27] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_59
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_59_params.batch_size, conv_59_params.in_dim, conv_59_params.in_channels,
        conv_59_params.out_channels, conv_59_params.out_dim,
        conv_59_params.stride, 1, conv_59_params.padding, conv_59_params.kernel_size,
        1024,

        (elem_t*)conv_58_out, (elem_t*)conv_59_w, ((acc_t*)conv_39_b + 544), ((elem_t*)conv_39_out_pooled + 544),

        RELU, conv_59_params.output_scale, 0,
        conv_59_params.pool_size, 0, conv_59_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 11
    // conv_60
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_60_params.I, conv_60_params.J, conv_60_params.K, 
            conv_60_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_60_w, (acc_t*)conv_60_b, (elem_t*)conv_60_out,
            RELU, conv_60_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[28] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_61
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_61_params.batch_size, conv_61_params.in_dim, conv_61_params.in_channels,
        conv_61_params.out_channels, conv_61_params.out_dim,
        conv_61_params.stride, 1, conv_61_params.padding, conv_61_params.kernel_size,
        1024,

        (elem_t*)conv_60_out, (elem_t*)conv_61_w, ((acc_t*)conv_39_b + 576), ((elem_t*)conv_39_out_pooled + 576),

        RELU, conv_61_params.output_scale, 0,
        conv_61_params.pool_size, 0, conv_61_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 12
    // conv_62
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_62_params.I, conv_62_params.J, conv_62_params.K, 
            conv_62_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_62_w, (acc_t*)conv_62_b, (elem_t*)conv_62_out,
            RELU, conv_62_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[29] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_63
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_63_params.batch_size, conv_63_params.in_dim, conv_63_params.in_channels,
        conv_63_params.out_channels, conv_63_params.out_dim,
        conv_63_params.stride, 1, conv_63_params.padding, conv_63_params.kernel_size,
        1024,

        (elem_t*)conv_62_out, (elem_t*)conv_63_w, ((acc_t*)conv_39_b + 608), ((elem_t*)conv_39_out_pooled + 608),

        RELU, conv_63_params.output_scale, 0,
        conv_63_params.pool_size, 0, conv_63_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 13
    // conv_64
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_64_params.I, conv_64_params.J, conv_64_params.K, 
            conv_64_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_64_w, (acc_t*)conv_64_b, (elem_t*)conv_64_out,
            RELU, conv_64_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[30] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_65
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_65_params.batch_size, conv_65_params.in_dim, conv_65_params.in_channels,
        conv_65_params.out_channels, conv_65_params.out_dim,
        conv_65_params.stride, 1, conv_65_params.padding, conv_65_params.kernel_size,
        1024,

        (elem_t*)conv_64_out, (elem_t*)conv_65_w, ((acc_t*)conv_39_b + 640), ((elem_t*)conv_39_out_pooled + 640),

        RELU, conv_65_params.output_scale, 0,
        conv_65_params.pool_size, 0, conv_65_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 14
    // conv_66
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_66_params.I, conv_66_params.J, conv_66_params.K, 
            conv_66_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_66_w, (acc_t*)conv_66_b, (elem_t*)conv_66_out,
            RELU, conv_66_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[31] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_67
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_67_params.batch_size, conv_67_params.in_dim, conv_67_params.in_channels,
        conv_67_params.out_channels, conv_67_params.out_dim,
        conv_67_params.stride, 1, conv_67_params.padding, conv_67_params.kernel_size,
        1024,

        (elem_t*)conv_66_out, (elem_t*)conv_67_w, ((acc_t*)conv_39_b + 672), ((elem_t*)conv_39_out_pooled + 672),

        RELU, conv_67_params.output_scale, 0,
        conv_67_params.pool_size, 0, conv_67_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 15
    // conv_68
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_68_params.I, conv_68_params.J, conv_68_params.K, 
            conv_68_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_68_w, (acc_t*)conv_68_b, (elem_t*)conv_68_out,
            RELU, conv_68_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[32] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_69
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_69_params.batch_size, conv_69_params.in_dim, conv_69_params.in_channels,
        conv_69_params.out_channels, conv_69_params.out_dim,
        conv_69_params.stride, 1, conv_69_params.padding, conv_69_params.kernel_size,
        1024,

        (elem_t*)conv_68_out, (elem_t*)conv_69_w, ((acc_t*)conv_39_b + 704), ((elem_t*)conv_39_out_pooled + 704),

        RELU, conv_69_params.output_scale, 0,
        conv_69_params.pool_size, 0, conv_69_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 16
    // conv_70
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_70_params.I, conv_70_params.J, conv_70_params.K, 
            conv_70_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_70_w, (acc_t*)conv_70_b, (elem_t*)conv_70_out,
            RELU, conv_70_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[33] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_71
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_71_params.batch_size, conv_71_params.in_dim, conv_71_params.in_channels,
        conv_71_params.out_channels, conv_71_params.out_dim,
        conv_71_params.stride, 1, conv_71_params.padding, conv_71_params.kernel_size,
        1024,

        (elem_t*)conv_70_out, (elem_t*)conv_71_w, ((acc_t*)conv_39_b + 736), ((elem_t*)conv_39_out_pooled + 736),

        RELU, conv_71_params.output_scale, 0,
        conv_71_params.pool_size, 0, conv_71_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 17
    // conv_72
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_72_params.I, conv_72_params.J, conv_72_params.K, 
            conv_72_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_72_w, (acc_t*)conv_72_b, (elem_t*)conv_72_out,
            RELU, conv_72_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[34] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_73
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_73_params.batch_size, conv_73_params.in_dim, conv_73_params.in_channels,
        conv_73_params.out_channels, conv_73_params.out_dim,
        conv_73_params.stride, 1, conv_73_params.padding, conv_73_params.kernel_size,
        1024,

        (elem_t*)conv_72_out, (elem_t*)conv_73_w, ((acc_t*)conv_39_b + 768), ((elem_t*)conv_39_out_pooled + 768),

        RELU, conv_73_params.output_scale, 0,
        conv_73_params.pool_size, 0, conv_73_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 18
    // conv_74
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_74_params.I, conv_74_params.J, conv_74_params.K, 
            conv_74_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_74_w, (acc_t*)conv_74_b, (elem_t*)conv_74_out,
            RELU, conv_74_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[35] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_75
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_75_params.batch_size, conv_75_params.in_dim, conv_75_params.in_channels,
        conv_75_params.out_channels, conv_75_params.out_dim,
        conv_75_params.stride, 1, conv_75_params.padding, conv_75_params.kernel_size,
        1024,

        (elem_t*)conv_74_out, (elem_t*)conv_75_w, ((acc_t*)conv_39_b + 800), ((elem_t*)conv_39_out_pooled + 800),

        RELU, conv_75_params.output_scale, 0,
        conv_75_params.pool_size, 0, conv_75_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 19
    // conv_76
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_76_params.I, conv_76_params.J, conv_76_params.K, 
            conv_76_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_76_w, (acc_t*)conv_76_b, (elem_t*)conv_76_out,
            RELU, conv_76_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[36] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_77
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_77_params.batch_size, conv_77_params.in_dim, conv_77_params.in_channels,
        conv_77_params.out_channels, conv_77_params.out_dim,
        conv_77_params.stride, 1, conv_77_params.padding, conv_77_params.kernel_size,
        1024,

        (elem_t*)conv_76_out, (elem_t*)conv_77_w, ((acc_t*)conv_39_b + 832), ((elem_t*)conv_39_out_pooled + 832),

        RELU, conv_77_params.output_scale, 0,
        conv_77_params.pool_size, 0, conv_77_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 20
    // conv_78
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_78_params.I, conv_78_params.J, conv_78_params.K, 
            conv_78_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_78_w, (acc_t*)conv_78_b, (elem_t*)conv_78_out,
            RELU, conv_78_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[37] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_79
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_79_params.batch_size, conv_79_params.in_dim, conv_79_params.in_channels,
        conv_79_params.out_channels, conv_79_params.out_dim,
        conv_79_params.stride, 1, conv_79_params.padding, conv_79_params.kernel_size,
        1024,

        (elem_t*)conv_78_out, (elem_t*)conv_79_w, ((acc_t*)conv_39_b + 864), ((elem_t*)conv_39_out_pooled + 864),

        RELU, conv_79_params.output_scale, 0,
        conv_79_params.pool_size, 0, conv_79_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 21
    // conv_80
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_80_params.I, conv_80_params.J, conv_80_params.K, 
            conv_80_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_80_w, (acc_t*)conv_80_b, (elem_t*)conv_80_out,
            RELU, conv_80_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[38] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_81
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_81_params.batch_size, conv_81_params.in_dim, conv_81_params.in_channels,
        conv_81_params.out_channels, conv_81_params.out_dim,
        conv_81_params.stride, 1, conv_81_params.padding, conv_81_params.kernel_size,
        1024,

        (elem_t*)conv_80_out, (elem_t*)conv_81_w, ((acc_t*)conv_39_b + 896), ((elem_t*)conv_39_out_pooled + 896),

        RELU, conv_81_params.output_scale, 0,
        conv_81_params.pool_size, 0, conv_81_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 22
    // conv_82
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_82_params.I, conv_82_params.J, conv_82_params.K, 
            conv_82_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_82_w, (acc_t*)conv_82_b, (elem_t*)conv_82_out,
            RELU, conv_82_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[39] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_83
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_83_params.batch_size, conv_83_params.in_dim, conv_83_params.in_channels,
        conv_83_params.out_channels, conv_83_params.out_dim,
        conv_83_params.stride, 1, conv_83_params.padding, conv_83_params.kernel_size,
        1024,

        (elem_t*)conv_82_out, (elem_t*)conv_83_w, ((acc_t*)conv_39_b + 928), ((elem_t*)conv_39_out_pooled + 928),

        RELU, conv_83_params.output_scale, 0,
        conv_83_params.pool_size, 0, conv_83_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 23
    // conv_84
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_84_params.I, conv_84_params.J, conv_84_params.K, 
            conv_84_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_84_w, (acc_t*)conv_84_b, (elem_t*)conv_84_out,
            RELU, conv_84_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[40] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_85
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_85_params.batch_size, conv_85_params.in_dim, conv_85_params.in_channels,
        conv_85_params.out_channels, conv_85_params.out_dim,
        conv_85_params.stride, 1, conv_85_params.padding, conv_85_params.kernel_size,
        1024,

        (elem_t*)conv_84_out, (elem_t*)conv_85_w, ((acc_t*)conv_39_b + 960), ((elem_t*)conv_39_out_pooled + 960),

        RELU, conv_85_params.output_scale, 0,
        conv_85_params.pool_size, 0, conv_85_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 24
    // conv_86
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_86_params.I, conv_86_params.J, conv_86_params.K, 
            conv_86_params.J, 1024,
            (elem_t*)conv_39_out_pooled, (elem_t*)conv_86_w, (acc_t*)conv_86_b, (elem_t*)conv_86_out,
            RELU, conv_86_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[41] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_87
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_87_params.batch_size, conv_87_params.in_dim, conv_87_params.in_channels,
        conv_87_params.out_channels, conv_87_params.out_dim,
        conv_87_params.stride, 1, conv_87_params.padding, conv_87_params.kernel_size,
        1024,

        (elem_t*)conv_86_out, (elem_t*)conv_87_w, ((acc_t*)conv_39_b + 992), ((elem_t*)conv_39_out_pooled + 992),

        RELU, conv_87_params.output_scale, 0,
        conv_87_params.pool_size, 0, conv_87_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // conv_88
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_88_params.batch_size, conv_88_params.in_dim, conv_88_params.in_channels,
        conv_88_params.out_channels, conv_88_params.out_dim,
        conv_88_params.stride, 1, conv_88_params.padding, conv_88_params.kernel_size,
        1024,

        (elem_t*)conv_39_out_pooled, (elem_t*)conv_88_w, ((acc_t*)conv_88_b + 0), ((elem_t*)conv_88_out_pooled + 0),

        RELU, conv_88_params.output_scale, 0,
        conv_88_params.pool_size, conv_88_params.pool_stride, conv_88_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_block 4
    // dense_layer 1
    // conv_89
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_89_params.I, conv_89_params.J, conv_89_params.K, 
            conv_89_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_89_w, (acc_t*)conv_89_b, (elem_t*)conv_89_out,
            RELU, conv_89_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[42] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_90
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_90_params.batch_size, conv_90_params.in_dim, conv_90_params.in_channels,
        conv_90_params.out_channels, conv_90_params.out_dim,
        conv_90_params.stride, 1, conv_90_params.padding, conv_90_params.kernel_size,
        1024,

        (elem_t*)conv_89_out, (elem_t*)conv_90_w, ((acc_t*)conv_88_b + 512), ((elem_t*)conv_88_out_pooled + 512),

        RELU, conv_90_params.output_scale, 0,
        conv_90_params.pool_size, 0, conv_90_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 2
    // conv_91
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_91_params.I, conv_91_params.J, conv_91_params.K, 
            conv_91_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_91_w, (acc_t*)conv_91_b, (elem_t*)conv_91_out,
            RELU, conv_91_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[43] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_92
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_92_params.batch_size, conv_92_params.in_dim, conv_92_params.in_channels,
        conv_92_params.out_channels, conv_92_params.out_dim,
        conv_92_params.stride, 1, conv_92_params.padding, conv_92_params.kernel_size,
        1024,

        (elem_t*)conv_91_out, (elem_t*)conv_92_w, ((acc_t*)conv_88_b + 544), ((elem_t*)conv_88_out_pooled + 544),

        RELU, conv_92_params.output_scale, 0,
        conv_92_params.pool_size, 0, conv_92_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 3
    // conv_93
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_93_params.I, conv_93_params.J, conv_93_params.K, 
            conv_93_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_93_w, (acc_t*)conv_93_b, (elem_t*)conv_93_out,
            RELU, conv_93_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[44] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_94
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_94_params.batch_size, conv_94_params.in_dim, conv_94_params.in_channels,
        conv_94_params.out_channels, conv_94_params.out_dim,
        conv_94_params.stride, 1, conv_94_params.padding, conv_94_params.kernel_size,
        1024,

        (elem_t*)conv_93_out, (elem_t*)conv_94_w, ((acc_t*)conv_88_b + 576), ((elem_t*)conv_88_out_pooled + 576),

        RELU, conv_94_params.output_scale, 0,
        conv_94_params.pool_size, 0, conv_94_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 4
    // conv_95
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_95_params.I, conv_95_params.J, conv_95_params.K, 
            conv_95_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_95_w, (acc_t*)conv_95_b, (elem_t*)conv_95_out,
            RELU, conv_95_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[45] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_96
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_96_params.batch_size, conv_96_params.in_dim, conv_96_params.in_channels,
        conv_96_params.out_channels, conv_96_params.out_dim,
        conv_96_params.stride, 1, conv_96_params.padding, conv_96_params.kernel_size,
        1024,

        (elem_t*)conv_95_out, (elem_t*)conv_96_w, ((acc_t*)conv_88_b + 608), ((elem_t*)conv_88_out_pooled + 608),

        RELU, conv_96_params.output_scale, 0,
        conv_96_params.pool_size, 0, conv_96_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 5
    // conv_97
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_97_params.I, conv_97_params.J, conv_97_params.K, 
            conv_97_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_97_w, (acc_t*)conv_97_b, (elem_t*)conv_97_out,
            RELU, conv_97_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[46] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_98
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_98_params.batch_size, conv_98_params.in_dim, conv_98_params.in_channels,
        conv_98_params.out_channels, conv_98_params.out_dim,
        conv_98_params.stride, 1, conv_98_params.padding, conv_98_params.kernel_size,
        1024,

        (elem_t*)conv_97_out, (elem_t*)conv_98_w, ((acc_t*)conv_88_b + 640), ((elem_t*)conv_88_out_pooled + 640),

        RELU, conv_98_params.output_scale, 0,
        conv_98_params.pool_size, 0, conv_98_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 6
    // conv_99
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_99_params.I, conv_99_params.J, conv_99_params.K, 
            conv_99_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_99_w, (acc_t*)conv_99_b, (elem_t*)conv_99_out,
            RELU, conv_99_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[47] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_100
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_100_params.batch_size, conv_100_params.in_dim, conv_100_params.in_channels,
        conv_100_params.out_channels, conv_100_params.out_dim,
        conv_100_params.stride, 1, conv_100_params.padding, conv_100_params.kernel_size,
        1024,

        (elem_t*)conv_99_out, (elem_t*)conv_100_w, ((acc_t*)conv_88_b + 672), ((elem_t*)conv_88_out_pooled + 672),

        RELU, conv_100_params.output_scale, 0,
        conv_100_params.pool_size, 0, conv_100_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 7
    // conv_101
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_101_params.I, conv_101_params.J, conv_101_params.K, 
            conv_101_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_101_w, (acc_t*)conv_101_b, (elem_t*)conv_101_out,
            RELU, conv_101_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[48] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_102
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_102_params.batch_size, conv_102_params.in_dim, conv_102_params.in_channels,
        conv_102_params.out_channels, conv_102_params.out_dim,
        conv_102_params.stride, 1, conv_102_params.padding, conv_102_params.kernel_size,
        1024,

        (elem_t*)conv_101_out, (elem_t*)conv_102_w, ((acc_t*)conv_88_b + 704), ((elem_t*)conv_88_out_pooled + 704),

        RELU, conv_102_params.output_scale, 0,
        conv_102_params.pool_size, 0, conv_102_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 8
    // conv_103
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_103_params.I, conv_103_params.J, conv_103_params.K, 
            conv_103_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_103_w, (acc_t*)conv_103_b, (elem_t*)conv_103_out,
            RELU, conv_103_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[49] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_104
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_104_params.batch_size, conv_104_params.in_dim, conv_104_params.in_channels,
        conv_104_params.out_channels, conv_104_params.out_dim,
        conv_104_params.stride, 1, conv_104_params.padding, conv_104_params.kernel_size,
        1024,

        (elem_t*)conv_103_out, (elem_t*)conv_104_w, ((acc_t*)conv_88_b + 736), ((elem_t*)conv_88_out_pooled + 736),

        RELU, conv_104_params.output_scale, 0,
        conv_104_params.pool_size, 0, conv_104_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 9
    // conv_105
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_105_params.I, conv_105_params.J, conv_105_params.K, 
            conv_105_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_105_w, (acc_t*)conv_105_b, (elem_t*)conv_105_out,
            RELU, conv_105_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[50] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_106
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_106_params.batch_size, conv_106_params.in_dim, conv_106_params.in_channels,
        conv_106_params.out_channels, conv_106_params.out_dim,
        conv_106_params.stride, 1, conv_106_params.padding, conv_106_params.kernel_size,
        1024,

        (elem_t*)conv_105_out, (elem_t*)conv_106_w, ((acc_t*)conv_88_b + 768), ((elem_t*)conv_88_out_pooled + 768),

        RELU, conv_106_params.output_scale, 0,
        conv_106_params.pool_size, 0, conv_106_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[54] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 10
    // conv_107
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_107_params.I, conv_107_params.J, conv_107_params.K, 
            conv_107_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_107_w, (acc_t*)conv_107_b, (elem_t*)conv_107_out,
            RELU, conv_107_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[51] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_108
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_108_params.batch_size, conv_108_params.in_dim, conv_108_params.in_channels,
        conv_108_params.out_channels, conv_108_params.out_dim,
        conv_108_params.stride, 1, conv_108_params.padding, conv_108_params.kernel_size,
        1024,

        (elem_t*)conv_107_out, (elem_t*)conv_108_w, ((acc_t*)conv_88_b + 800), ((elem_t*)conv_88_out_pooled + 800),

        RELU, conv_108_params.output_scale, 0,
        conv_108_params.pool_size, 0, conv_108_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[55] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 11
    // conv_109
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_109_params.I, conv_109_params.J, conv_109_params.K, 
            conv_109_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_109_w, (acc_t*)conv_109_b, (elem_t*)conv_109_out,
            RELU, conv_109_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[52] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_110
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_110_params.batch_size, conv_110_params.in_dim, conv_110_params.in_channels,
        conv_110_params.out_channels, conv_110_params.out_dim,
        conv_110_params.stride, 1, conv_110_params.padding, conv_110_params.kernel_size,
        1024,

        (elem_t*)conv_109_out, (elem_t*)conv_110_w, ((acc_t*)conv_88_b + 832), ((elem_t*)conv_88_out_pooled + 832),

        RELU, conv_110_params.output_scale, 0,
        conv_110_params.pool_size, 0, conv_110_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[56] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 12
    // conv_111
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_111_params.I, conv_111_params.J, conv_111_params.K, 
            conv_111_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_111_w, (acc_t*)conv_111_b, (elem_t*)conv_111_out,
            RELU, conv_111_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[53] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_112
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_112_params.batch_size, conv_112_params.in_dim, conv_112_params.in_channels,
        conv_112_params.out_channels, conv_112_params.out_dim,
        conv_112_params.stride, 1, conv_112_params.padding, conv_112_params.kernel_size,
        1024,

        (elem_t*)conv_111_out, (elem_t*)conv_112_w, ((acc_t*)conv_88_b + 864), ((elem_t*)conv_88_out_pooled + 864),

        RELU, conv_112_params.output_scale, 0,
        conv_112_params.pool_size, 0, conv_112_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[57] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 13
    // conv_113
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_113_params.I, conv_113_params.J, conv_113_params.K, 
            conv_113_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_113_w, (acc_t*)conv_113_b, (elem_t*)conv_113_out,
            RELU, conv_113_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[54] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_114
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_114_params.batch_size, conv_114_params.in_dim, conv_114_params.in_channels,
        conv_114_params.out_channels, conv_114_params.out_dim,
        conv_114_params.stride, 1, conv_114_params.padding, conv_114_params.kernel_size,
        1024,

        (elem_t*)conv_113_out, (elem_t*)conv_114_w, ((acc_t*)conv_88_b + 896), ((elem_t*)conv_88_out_pooled + 896),

        RELU, conv_114_params.output_scale, 0,
        conv_114_params.pool_size, 0, conv_114_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[58] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 14
    // conv_115
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_115_params.I, conv_115_params.J, conv_115_params.K, 
            conv_115_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_115_w, (acc_t*)conv_115_b, (elem_t*)conv_115_out,
            RELU, conv_115_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[55] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_116
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_116_params.batch_size, conv_116_params.in_dim, conv_116_params.in_channels,
        conv_116_params.out_channels, conv_116_params.out_dim,
        conv_116_params.stride, 1, conv_116_params.padding, conv_116_params.kernel_size,
        1024,

        (elem_t*)conv_115_out, (elem_t*)conv_116_w, ((acc_t*)conv_88_b + 928), ((elem_t*)conv_88_out_pooled + 928),

        RELU, conv_116_params.output_scale, 0,
        conv_116_params.pool_size, 0, conv_116_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[59] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 15
    // conv_117
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_117_params.I, conv_117_params.J, conv_117_params.K, 
            conv_117_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_117_w, (acc_t*)conv_117_b, (elem_t*)conv_117_out,
            RELU, conv_117_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[56] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_118
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_118_params.batch_size, conv_118_params.in_dim, conv_118_params.in_channels,
        conv_118_params.out_channels, conv_118_params.out_dim,
        conv_118_params.stride, 1, conv_118_params.padding, conv_118_params.kernel_size,
        1024,

        (elem_t*)conv_117_out, (elem_t*)conv_118_w, ((acc_t*)conv_88_b + 960), ((elem_t*)conv_88_out_pooled + 960),

        RELU, conv_118_params.output_scale, 0,
        conv_118_params.pool_size, 0, conv_118_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[60] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // dense_layer 16
    // conv_119
    start = read_cycles();
        tiled_matmul_nn_auto_in_stride(conv_119_params.I, conv_119_params.J, conv_119_params.K, 
            conv_119_params.J, 1024,
            (elem_t*)conv_88_out_pooled, (elem_t*)conv_119_w, (acc_t*)conv_119_b, (elem_t*)conv_119_out,
            RELU, conv_119_params.output_scale, 0, true,
            tiled_matmul_type,
            OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->matmul_cycles[57] = end - start;
    #if THREAD_SYNC == 1
        pthread_barrier_wait(&barrier);
    #endif
        
    // conv_120
     start = read_cycles();

    tiled_conv_A_stride_auto_bubble(
        conv_120_params.batch_size, conv_120_params.in_dim, conv_120_params.in_channels,
        conv_120_params.out_channels, conv_120_params.out_dim,
        conv_120_params.stride, 1, conv_120_params.padding, conv_120_params.kernel_size,
        1024,

        (elem_t*)conv_119_out, (elem_t*)conv_120_w, ((acc_t*)conv_88_b + 992), ((elem_t*)conv_88_out_pooled + 992),

        RELU, conv_120_params.output_scale, 0,
        conv_120_params.pool_size, 0, conv_120_params.pool_padding, false,

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_IMAGE_DEN121, SKIP_WEIGHT_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[61] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif  


    // Global averaging
    
    static elem_t average[1][1024] row_align(MAX_BLOCK_LEN);

    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_88_out_pooled, average, conv_88_params.batch_size,                         
            conv_88_params.out_channels, conv_88_params.out_dim, WS);
       

    end = read_cycles();
    nn_args->other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif

    // fc_121
    start = read_cycles();

    tiled_matmul_nn_auto_bubble(fc_121_params.I, fc_121_params.J, fc_121_params.K, fc_121_params.J,
        (elem_t*)average, (elem_t*)fc_121_w, (acc_t*)fc_121_b, (elem_t*)fc_121_out,
        NO_ACTIVATION, fc_121_params.output_scale, 0, false, tiled_matmul_type,
        OROW_DIVIDE, BATCH_DIVIDE, cid, SKIP_A_DEN121, SKIP_B_DEN121, PROFILE_DEN121, LATENCY_DEN121, ALERT_CYCLE_DEN121, UNLOCK_CYCLE_DEN121, PAUSE_TURN_DEN121);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[58] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif   


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

    if (pthread_mutex_init(&dense121_mutex, NULL) != 0)
    {
        printf("\n mutex init failed\n");
        return 1;
    }

    pthread_barrier_init(&barrier, NULL, OROW_DIVIDE);
    uint64_t start = read_cycles();
    for(int i = 0; i < OROW_DIVIDE; i++)
        pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
    
    for(int i = 0; i < OROW_DIVIDE; i++)
        pthread_join(thread[i], NULL);
    uint64_t end = read_cycles();
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&dense121_mutex);

    printf("total cycles with threading overhead: %llu \\n", end - start);

    for(int i = 0; i < OROW_DIVIDE; i++){
        uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
        uint64_t conv_cycles = nn_args[i].total_conv_cycles;
        uint64_t res_add_cycles = nn_args[i].total_resadd_cycles;
        uint64_t other_cycles = nn_args[i].other_cycles;
        uint64_t total_cycles =  conv_cycles + matmul_cycles + res_add_cycles + other_cycles;
        uint64_t thread_cycles = nn_args[i].total_thread_cycles;


        printf("\nproc %d total thread cycles: %llu\n", i, thread_cycles);
        printf("proc %d Total cycles: %llu (100%%)\n", i, total_cycles);
        printf("proc %d Matmul cycles: %llu (%d%%)\n", i, matmul_cycles, (matmul_cycles * 100) / total_cycles);
        printf("proc %d Conv cycles: %llu (%d%%)\n", i, conv_cycles, (conv_cycles * 100) / total_cycles);
        //printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
        printf("proc %d Res add cycles: %llu (%d%%)\n", i, res_add_cycles, (res_add_cycles * 100) / total_cycles);
        printf("proc %d Other cycles: %llu (%d%%)\n", i, other_cycles, (other_cycles * 100) / total_cycles);
        for(int j = 0; j < 62; j++)
            printf("conv layer %d cycles: %llu \n", j, nn_args[i].conv_cycles[j]);
        for(int j = 0; j < 59; j++)
            printf("matmul layer %d cycles: %llu \n", j, nn_args[i].matmul_cycles[j]);
        for(int j = 0; j < 0; j++)
            printf("resadd %d cycles: %llu \n", j, nn_args[i].res_add_cycles[j]);
        printf("==================================\n");
        

    }
        printf("worst case for each layers \n");
    

    for(int i = 0; i < 62; i++)    

    {
        uint64_t max = 0;
        for(int j = 0; j < OROW_DIVIDE; j++)
           max = (max > nn_args[j].conv_cycles[i]) ? max : nn_args[j].conv_cycles[i];
        
        printf("conv layer %d worst cycles: %llu \n", i, max);
        max = 0;
    }
    

    for(int i = 0; i < 59; i++)    

    {
        uint64_t max = 0;
        for(int j = 0; j < OROW_DIVIDE; j++)
           max = (max > nn_args[j].matmul_cycles[i]) ? max : nn_args[j].matmul_cycles[i];
        
        printf("matmul layer %d worst cycles: %llu \n", i, max);
        max = 0;
    

    }
    

    for(int i = 0; i < 0; i++)    

    {
        uint64_t max = 0;
        for(int j = 0; j < OROW_DIVIDE; j++)
           max = (max > nn_args[j].res_add_cycles[i]) ? max : nn_args[j].res_add_cycles[i];
        
        printf("res_add layer %d worst cycles: %llu \n", i, max);
        max = 0;

        

    }
    printf("==================================\\n");
    exit(0);
}

