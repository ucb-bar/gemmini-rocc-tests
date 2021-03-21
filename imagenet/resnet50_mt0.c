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

#include "resnet50_mt_params.h"
#include "images.h"

#define num_layer 54
#define num_resadd 16

#define OCH_DIVIDE 1 // 1: independent, 2: 2+2 collab, 4: sequential
#define OCH_PADDING false
#define ICH_PADDING false
#define A_PADDING ICH_PADDING
#define B_PADDING OCH_PADDING
#define SKIP_A false
#define SKIP_B false
#define SKIP_WEIGHT false //later

pthread_barrier_t barrier;

//meaningless
static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t Out[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

struct thread_args{
    uint64_t total_conv_cycles, total_matmul_cycles, total_resadd_cycles;
	uint64_t res_add_cycles[num_resadd];
	uint64_t conv_cycles[num_layer]; //including final FC layer
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
	gemmini_flush(0);
    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;
    
    pthread_barrier_wait(&barrier);
    
    // conv_1
    {
        start = read_cycles();
    //compute image address
        tiled_conv_A_stride_auto_loopld(
            conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_dim,
            conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale, 0,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[0] = end - start;

    }

    // conv_2
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[1] = end - start;
    }

    // conv_3
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_dim,
            conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

            RELU, conv_3_params.output_scale, 0,
            conv_3_params.pool_size, 0, conv_3_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[2] = end - start;
   }

    // conv_4
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            NO_ACTIVATION, conv_4_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[3] = end - start;
    }

    // Downsampling conv_1_out_pooled
    // conv_5
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_1_out_pooled, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, 0, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[4] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[0] = end - start;

    // conv_6
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_4_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[5] = end - start;

   }

    // conv_7
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
            conv_7_params.out_channels, conv_7_params.out_dim,
            conv_7_params.stride, 1, conv_7_params.padding, conv_7_params.kernel_size,

            (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

            RELU, conv_7_params.output_scale, 0,
            conv_7_params.pool_size, 0, conv_7_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[6] = end - start;

   }

    // conv_8
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out, conv_8_w, conv_8_b, conv_8_out,
            NO_ACTIVATION, conv_8_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[7] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[1] = end - start;


    // conv_9
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[8] = end - start;

   }

    // conv_10
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_10_params.batch_size, conv_10_params.in_dim, conv_10_params.in_channels,
            conv_10_params.out_channels, conv_10_params.out_dim,
            conv_10_params.stride, 1, conv_10_params.padding, conv_10_params.kernel_size,

            (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

            RELU, conv_10_params.output_scale, 0,
            conv_10_params.pool_size, 0, conv_10_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[9] = end - start;

   }

    // conv_11
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            NO_ACTIVATION, conv_11_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[10] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[2] = end - start;


    // conv_12
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[11] = end - start;

   }

    // conv_13
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_dim,
            conv_13_params.stride, 1, conv_13_params.padding, conv_13_params.kernel_size,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

            RELU, conv_13_params.output_scale, 0,
            conv_13_params.pool_size, 0, conv_13_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[12] = end - start;

   }

    // conv_14
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out, conv_14_w, conv_14_b, conv_14_out,
            NO_ACTIVATION, conv_14_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[13] = end - start;

   }

    // Downsampling conv_11_out
    // conv_15
    {
        start = read_cycles();

        // tiled_conv_A_stride_auto_loopld(
        tiled_conv_downsample_loopld(
            conv_15_params.batch_size, conv_15_params.in_dim, conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_dim,
            // conv_15_params.stride, 1, conv_15_params.padding, conv_15_params.kernel_size,

            (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

            NO_ACTIVATION, conv_15_params.output_scale, 0,
            // conv_15_params.pool_size, 0, conv_15_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[14] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[3] = end - start;

    // conv_16
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_14_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[15] = end - start;

   }

    // conv_17
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_17_params.batch_size, conv_17_params.in_dim, conv_17_params.in_channels,
            conv_17_params.out_channels, conv_17_params.out_dim,
            conv_17_params.stride, 1, conv_17_params.padding, conv_17_params.kernel_size,

            (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

            RELU, conv_17_params.output_scale, 0,
            conv_17_params.pool_size, 0, conv_17_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[16] = end - start;

   }

    // conv_18
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[17] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[4] = end - start;

    // conv_19
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[18] = end - start;

   }

    // conv_20
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_20_params.batch_size, conv_20_params.in_dim, conv_20_params.in_channels,
            conv_20_params.out_channels, conv_20_params.out_dim,
            conv_20_params.stride, 1, conv_20_params.padding, conv_20_params.kernel_size,

            (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

            RELU, conv_20_params.output_scale, 0,
            conv_20_params.pool_size, 0, conv_20_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[19] = end - start;

   }

    // conv_21
   {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
       nn_args->conv_cycles[20] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[5] = end - start;

    // conv_22
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[21] = end - start;

   }

    // conv_23
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_23_params.batch_size, conv_23_params.in_dim, conv_23_params.in_channels,
            conv_23_params.out_channels, conv_23_params.out_dim,
            conv_23_params.stride, 1, conv_23_params.padding, conv_23_params.kernel_size,

            (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

            RELU, conv_23_params.output_scale, 0,
            conv_23_params.pool_size, 0, conv_23_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[22] = end - start;

   }

    // conv_24
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[23] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[6] = end - start;

    // conv_25
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[24] = end - start;

   }

    // conv_26
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_26_params.batch_size, conv_26_params.in_dim, conv_26_params.in_channels,
            conv_26_params.out_channels, conv_26_params.out_dim,
            conv_26_params.stride, 1, conv_26_params.padding, conv_26_params.kernel_size,

            (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

            RELU, conv_26_params.output_scale, 0,
            conv_26_params.pool_size, 0, conv_26_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[25] = end - start;

   }

    // conv_27
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[26] = end - start;

   }

    // Downsampling conv_24_out
    // conv_28
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
        //tiled_conv_downsample_loopld(
            conv_28_params.batch_size, conv_28_params.in_dim, conv_28_params.in_channels,
            conv_28_params.out_channels, conv_28_params.out_dim,
            conv_28_params.stride, 1, conv_28_params.padding, conv_28_params.kernel_size,

            (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,

            NO_ACTIVATION, conv_28_params.output_scale, 0,
            conv_28_params.pool_size, 0, conv_28_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[27] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[7] = end - start;

    // conv_29
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_27_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[28] = end - start;

   }

    // conv_30
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_dim,
            conv_30_params.stride, 1, conv_30_params.padding, conv_30_params.kernel_size,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

            RELU, conv_30_params.output_scale, 0,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[29] = end - start;

   }

    // conv_31
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            NO_ACTIVATION, conv_31_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[30] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[8] = end - start;

    // conv_32
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_31_out, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[31] = end - start;

   }

    // conv_33
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_33_params.batch_size, conv_33_params.in_dim, conv_33_params.in_channels,
            conv_33_params.out_channels, conv_33_params.out_dim,
            conv_33_params.stride, 1, conv_33_params.padding, conv_33_params.kernel_size,

            (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

            RELU, conv_33_params.output_scale, 0,
            conv_33_params.pool_size, 0, conv_33_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[32] = end - start;

    }

    // conv_34
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            NO_ACTIVATION, conv_34_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[33] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[9] = end - start;

    // conv_35
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[34] = end - start;

   }

    // conv_36
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_36_params.batch_size, conv_36_params.in_dim, conv_36_params.in_channels,
            conv_36_params.out_channels, conv_36_params.out_dim,
            conv_36_params.stride, 1, conv_36_params.padding, conv_36_params.kernel_size,

            (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

            RELU, conv_36_params.output_scale, 0,
            conv_36_params.pool_size, 0, conv_36_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[35] = end - start;

   }

    // conv_37
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            NO_ACTIVATION, conv_37_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[36] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[10] = end - start;

    // conv_38
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_37_out, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[37] = end - start;

   }

    // conv_39
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_39_params.batch_size, conv_39_params.in_dim, conv_39_params.in_channels,
            conv_39_params.out_channels, conv_39_params.out_dim,
            conv_39_params.stride, 1, conv_39_params.padding, conv_39_params.kernel_size,

            (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

            RELU, conv_39_params.output_scale, 0,
            conv_39_params.pool_size, 0, conv_39_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[38] = end - start;

   }

    // conv_40
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            NO_ACTIVATION, conv_40_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[39] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[11] = end - start;

    // conv_41
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[40] = end - start;

   }

    // conv_42
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_42_params.batch_size, conv_42_params.in_dim, conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_dim,
            conv_42_params.stride, 1, conv_42_params.padding, conv_42_params.kernel_size,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

            RELU, conv_42_params.output_scale, 0,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[41] = end - start;

   }

    // conv_43
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            NO_ACTIVATION, conv_43_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[42] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[12] = end - start;

    // conv_44
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_43_out, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[43] = end - start;

   }

    // conv_45
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_45_params.batch_size, conv_45_params.in_dim, conv_45_params.in_channels,
            conv_45_params.out_channels, conv_45_params.out_dim,
            conv_45_params.stride, 1, conv_45_params.padding, conv_45_params.kernel_size,

            (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

            RELU, conv_45_params.output_scale, 0,
            conv_45_params.pool_size, 0, conv_45_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[44] = end - start;

   }

    // conv_46
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            NO_ACTIVATION, conv_46_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[45] = end - start;

   }

    // Downsampling conv_43_out
    // conv_47
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
        //tiled_conv_downsample_loopld(
            conv_47_params.batch_size, conv_47_params.in_dim, conv_47_params.in_channels,
            conv_47_params.out_channels, conv_47_params.out_dim,
            conv_47_params.stride, 1, conv_47_params.padding, conv_47_params.kernel_size,

            (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,

            NO_ACTIVATION, conv_47_params.output_scale, 0,
            conv_47_params.pool_size, 0, conv_47_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[46] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[13] = end - start;

    // conv_48
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_46_out, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[47] = end - start;

   }

    // conv_49
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_49_params.batch_size, conv_49_params.in_dim, conv_49_params.in_channels,
            conv_49_params.out_channels, conv_49_params.out_dim,
            conv_49_params.stride, 1, conv_49_params.padding, conv_49_params.kernel_size,

            (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

            RELU, conv_49_params.output_scale, 0,
            conv_49_params.pool_size, 0, conv_49_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[48] = end - start;

   }

    // conv_50
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_49_out, conv_50_w, conv_50_b, conv_50_out,
            NO_ACTIVATION, conv_50_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[49] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[14] = end - start;

    // conv_51
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld_loopld(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[50] = end - start;

   }

    // conv_52
    {
        start = read_cycles();

        tiled_conv_A_stride_auto_loopld(
            conv_52_params.batch_size, conv_52_params.in_dim, conv_52_params.in_channels,
            conv_52_params.out_channels, conv_52_params.out_dim,
            conv_52_params.stride, 1, conv_52_params.padding, conv_52_params.kernel_size,

            (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

            RELU, conv_52_params.output_scale, 0,
            conv_52_params.pool_size, 0, conv_52_params.pool_padding,

            WS, ICH_PADDING, OCH_PADDING, OCH_DIVIDE, false);

        end = read_cycles();
        conv_cycles += end - start;
        nn_args->conv_cycles[51] = end - start;

   }

    // conv_53
    {
        start = read_cycles();

        tiled_matmul_nn_auto_loopld(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            NO_ACTIVATION, conv_53_params.output_scale, 0, true,
            tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

        end = read_cycles();
        matmul_cycles += end - start;
        nn_args->conv_cycles[52] = end - start;

   }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    nn_args->res_add_cycles[15] = end - start;

    // Global averaging
    static elem_t average[2048][4] row_align(1);

    start = read_cycles();

    for (int batch = 0; batch < conv_53_params.batch_size; batch++) {
        for (int channel = 0; channel < conv_53_params.out_channels; channel++) {
            int sum = 0;
            for (int row = 0; row < conv_53_params.out_dim; row++) {
                for (int col = 0; col < conv_53_params.out_dim; col++) {
                    size_t r = batch * conv_53_params.out_dim * conv_53_params.out_dim + row * conv_53_params.out_dim + col;

                    sum += conv_53_out[r][channel];
                }
            }
            const int count = conv_53_params.out_dim * conv_53_params.out_dim;

            average[channel][batch] = (sum + count/2) / count;
        }
    }

    end = read_cycles();
    other_cycles += end - start;

    // fc_54
    start = read_cycles();

    tiled_matmul_nn_auto_loopld(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        fc_54_w, average, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, 0, false,
        tiled_matmul_type, SKIP_A, SKIP_B, A_PADDING, B_PADDING);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->conv_cycles[53] = end - start;

    nn_args->total_conv_cycles = conv_cycles;
    nn_args->total_matmul_cycles = matmul_cycles;
    nn_args->total_resadd_cycles = res_add_cycles;
    nn_args->other_cycles = other_cycles;

}

void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
   // char *msg;
   // msg = (char *) ptr;
    printf("print msg - cpu_id: %d \n", cpu_id);
   // printf("%s \n", msg);
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
        pthread_create(&thread[i], &attr[i], thread_matmul0, &matmul_args[i]);
    }

    for(int i = 0; i < num_proc; i++)
        pthread_join(thread[i], NULL);

    pthread_barrier_init(&barrier, NULL, num_proc);
    for(int i = 0; i < num_proc, i++)
        pthread_create(&thread[i], &attr[i], thread_NN, &thread_args[i]);
    
    for(int i = 0; i < num_proc; i++)
        pthread_join(thread[i], NULL);
    pthread_barrier_destroy(&barrier);
    


    for(int i = 0; i < num_proc, i++){
        uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
        uint64_t conv_cycles = nn_args[i].total_conv_cycles;
        uint64_t res_add_cycles = nn_args[i].total_resadd_cycles;
        uint64_t other_cycles = nn_args[i].other_cycles'
        uint64_t total_cycles =  conv_cycles + matmul_cycles + res_add_cycles + other_cycles;

        printf("\nproc %d Total cycles: %llu (100%%)\n", i, total_cycles);
        printf("proc %d Matmul cycles: %llu (%d%%)\n", i, matmul_cycles, (matmul_cycles * 100) / total_cycles);
        printf("proc %d Conv cycles: %llu (%d%%)\n", i, conv_cycles, (conv_cycles * 100) / total_cycles);
        //printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
        printf("proc %d Res add cycles: %llu (%d%%)\n", i, res_add_cycles, (res_add_cycles * 100) / total_cycles);
        printf("proc %d Other cycles: %llu (%d%%)\n", i, other_cycles, (other_cycles * 100) / total_cycles);
        for(int j = 0; j < num_layer; j++)
            printf("layer %d cycles: %llu \n", j, nn_args[i]->(conv_cycles[j]));
        for(int j = 0; j < num_resadd; j++)
            printf("resadd %d cycles: %llu \n", j, nn_args[i]->(res_add_cycles[j]));
        printf("==================================\n")
    }

    exit(0);
}

