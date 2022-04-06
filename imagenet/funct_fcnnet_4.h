
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "fcnnet_params_4.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif


#ifndef BAREMETAL
uint64_t* fcnnet_function_4(size_t cid, size_t group_id, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_fcnnet){
#else
uint64_t* fcnnet_function_4(size_t cid, size_t group_id, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (19+36+16+3)

  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[55];
    //uint64_t conv_cycles[36];
    uint64_t resadd_cycles[16];

 #if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif      
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_fcn4.batch_size, conv_1_params_fcn4.in_dim, conv_1_params_fcn4.in_channels,
        conv_1_params_fcn4.out_channels, conv_1_params_fcn4.out_dim,
        conv_1_params_fcn4.stride, conv_1_params_fcn4.dilation, conv_1_params_fcn4.padding, conv_1_params_fcn4.kernel_size,
        conv_1_params_fcn4.out_stride,

        (elem_t*)image4_s, (elem_t*)conv_1_w_fcn4, (acc_t*)conv_1_b_fcn4, (elem_t*)conv_1_out_fcn4_pooled,

        RELU, conv_1_params_fcn4.output_scale, 0,
        conv_1_params_fcn4.pool_size, conv_1_params_fcn4.pool_stride, conv_1_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params_fcn4.I, conv_2_params_fcn4.J, conv_2_params_fcn4.K, conv_2_params_fcn4.out_stride,
        (elem_t*)conv_1_out_fcn4_pooled, (elem_t*)conv_2_w_fcn4, (acc_t*)conv_2_b_fcn4, (elem_t*)conv_2_out_fcn4,
        NO_ACTIVATION, conv_2_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_3_params_fcn4.batch_size, conv_3_params_fcn4.in_dim, conv_3_params_fcn4.in_channels,
        conv_3_params_fcn4.out_channels, conv_3_params_fcn4.out_dim,
        conv_3_params_fcn4.stride, conv_3_params_fcn4.dilation, conv_3_params_fcn4.padding, conv_3_params_fcn4.kernel_size,
        conv_3_params_fcn4.out_stride,

        (elem_t*)conv_2_out_fcn4, (elem_t*)conv_3_w_fcn4, (acc_t*)conv_3_b_fcn4, (elem_t*)conv_3_out_fcn4,

        NO_ACTIVATION, conv_3_params_fcn4.output_scale, 0,
        conv_3_params_fcn4.pool_size, 0, conv_3_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_4
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_4_params_fcn4.I, conv_4_params_fcn4.J, conv_4_params_fcn4.K, conv_4_params_fcn4.out_stride,
        (elem_t*)conv_3_out_fcn4, (elem_t*)conv_4_w_fcn4, (acc_t*)conv_4_b_fcn4, (elem_t*)conv_4_out_fcn4,
        RELU, conv_4_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Downsampling conv_1_out_fcn4_pooled
    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params_fcn4.I, conv_5_params_fcn4.J, conv_5_params_fcn4.K, conv_5_params_fcn4.out_stride,
        (elem_t*)conv_1_out_fcn4_pooled, (elem_t*)conv_5_w_fcn4, (acc_t*)conv_5_b_fcn4, (elem_t*)conv_5_out_fcn4,
        NO_ACTIVATION, conv_5_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_4_params_fcn4.I, conv_4_params_fcn4.J,
        conv_4_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_5_out_fcn4,
        (elem_t*)conv_4_out_fcn4,
        (elem_t*)conv_4_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_6
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_6_params_fcn4.I, conv_6_params_fcn4.J, conv_6_params_fcn4.K, conv_6_params_fcn4.out_stride,
        (elem_t*)conv_4_out_fcn4, (elem_t*)conv_6_w_fcn4, (acc_t*)conv_6_b_fcn4, (elem_t*)conv_6_out_fcn4,
        NO_ACTIVATION, conv_6_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_7_params_fcn4.batch_size, conv_7_params_fcn4.in_dim, conv_7_params_fcn4.in_channels,
        conv_7_params_fcn4.out_channels, conv_7_params_fcn4.out_dim,
        conv_7_params_fcn4.stride, conv_7_params_fcn4.dilation, conv_7_params_fcn4.padding, conv_7_params_fcn4.kernel_size,
        conv_7_params_fcn4.out_stride,

        (elem_t*)conv_6_out_fcn4, (elem_t*)conv_7_w_fcn4, (acc_t*)conv_7_b_fcn4, (elem_t*)conv_7_out_fcn4,

        NO_ACTIVATION, conv_7_params_fcn4.output_scale, 0,
        conv_7_params_fcn4.pool_size, 0, conv_7_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params_fcn4.I, conv_8_params_fcn4.J, conv_8_params_fcn4.K, conv_8_params_fcn4.out_stride,
        (elem_t*)conv_7_out_fcn4, (elem_t*)conv_8_w_fcn4, (acc_t*)conv_8_b_fcn4, (elem_t*)conv_8_out_fcn4,
        RELU, conv_8_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_8_params_fcn4.I, conv_8_params_fcn4.J,
        conv_8_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_4_out_fcn4,
        (elem_t*)conv_8_out_fcn4,
        (elem_t*)conv_8_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params_fcn4.I, conv_9_params_fcn4.J, conv_9_params_fcn4.K, conv_9_params_fcn4.out_stride,
        (elem_t*)conv_8_out_fcn4, (elem_t*)conv_9_w_fcn4, (acc_t*)conv_9_b_fcn4, (elem_t*)conv_9_out_fcn4,
        NO_ACTIVATION, conv_9_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_10
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_10_params_fcn4.batch_size, conv_10_params_fcn4.in_dim, conv_10_params_fcn4.in_channels,
        conv_10_params_fcn4.out_channels, conv_10_params_fcn4.out_dim,
        conv_10_params_fcn4.stride, conv_10_params_fcn4.dilation, conv_10_params_fcn4.padding, conv_10_params_fcn4.kernel_size,
        conv_10_params_fcn4.out_stride,

        (elem_t*)conv_9_out_fcn4, (elem_t*)conv_10_w_fcn4, (acc_t*)conv_10_b_fcn4, (elem_t*)conv_10_out_fcn4,

        NO_ACTIVATION, conv_10_params_fcn4.output_scale, 0,
        conv_10_params_fcn4.pool_size, 0, conv_10_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params_fcn4.I, conv_11_params_fcn4.J, conv_11_params_fcn4.K, conv_11_params_fcn4.out_stride,
        (elem_t*)conv_10_out_fcn4, (elem_t*)conv_11_w_fcn4, (acc_t*)conv_11_b_fcn4, (elem_t*)conv_11_out_fcn4,
        RELU, conv_11_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_11_params_fcn4.I, conv_11_params_fcn4.J,
        conv_11_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_8_out_fcn4,
        (elem_t*)conv_11_out_fcn4,
        (elem_t*)conv_11_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_12
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_12_params_fcn4.I, conv_12_params_fcn4.J, conv_12_params_fcn4.K, conv_12_params_fcn4.out_stride,
        (elem_t*)conv_11_out_fcn4, (elem_t*)conv_12_w_fcn4, (acc_t*)conv_12_b_fcn4, (elem_t*)conv_12_out_fcn4,
        NO_ACTIVATION, conv_12_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_13
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_13_params_fcn4.batch_size, conv_13_params_fcn4.in_dim, conv_13_params_fcn4.in_channels,
        conv_13_params_fcn4.out_channels, conv_13_params_fcn4.out_dim,
        conv_13_params_fcn4.stride, conv_13_params_fcn4.dilation, conv_13_params_fcn4.padding, conv_13_params_fcn4.kernel_size,
        conv_13_params_fcn4.out_stride,

        (elem_t*)conv_12_out_fcn4, (elem_t*)conv_13_w_fcn4, (acc_t*)conv_13_b_fcn4, (elem_t*)conv_13_out_fcn4,

        NO_ACTIVATION, conv_13_params_fcn4.output_scale, 0,
        conv_13_params_fcn4.pool_size, 0, conv_13_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params_fcn4.I, conv_14_params_fcn4.J, conv_14_params_fcn4.K, conv_14_params_fcn4.out_stride,
        (elem_t*)conv_13_out_fcn4, (elem_t*)conv_14_w_fcn4, (acc_t*)conv_14_b_fcn4, (elem_t*)conv_14_out_fcn4,
        RELU, conv_14_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Downsampling conv_11_out_fcn4
    // conv_15
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_15_params_fcn4.batch_size, conv_15_params_fcn4.in_dim, conv_15_params_fcn4.in_channels,
        conv_15_params_fcn4.out_channels, conv_15_params_fcn4.out_dim,
        conv_15_params_fcn4.stride, conv_15_params_fcn4.dilation, conv_15_params_fcn4.padding, conv_15_params_fcn4.kernel_size,
        conv_15_params_fcn4.out_stride,

        (elem_t*)conv_11_out_fcn4, (elem_t*)conv_15_w_fcn4, (acc_t*)conv_15_b_fcn4, (elem_t*)conv_15_out_fcn4,

        NO_ACTIVATION, conv_15_params_fcn4.output_scale, 0,
        conv_15_params_fcn4.pool_size, 0, conv_15_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_14_params_fcn4.I, conv_14_params_fcn4.J,
        conv_14_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_15_out_fcn4,
        (elem_t*)conv_14_out_fcn4,
        (elem_t*)conv_14_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_16
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_16_params_fcn4.I, conv_16_params_fcn4.J, conv_16_params_fcn4.K, conv_16_params_fcn4.out_stride,
        (elem_t*)conv_14_out_fcn4, (elem_t*)conv_16_w_fcn4, (acc_t*)conv_16_b_fcn4, (elem_t*)conv_16_out_fcn4,
        NO_ACTIVATION, conv_16_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_17_params_fcn4.batch_size, conv_17_params_fcn4.in_dim, conv_17_params_fcn4.in_channels,
        conv_17_params_fcn4.out_channels, conv_17_params_fcn4.out_dim,
        conv_17_params_fcn4.stride, conv_17_params_fcn4.dilation, conv_17_params_fcn4.padding, conv_17_params_fcn4.kernel_size,
        conv_17_params_fcn4.out_stride,

        (elem_t*)conv_16_out_fcn4, (elem_t*)conv_17_w_fcn4, (acc_t*)conv_17_b_fcn4, (elem_t*)conv_17_out_fcn4,

        NO_ACTIVATION, conv_17_params_fcn4.output_scale, 0,
        conv_17_params_fcn4.pool_size, 0, conv_17_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params_fcn4.I, conv_18_params_fcn4.J, conv_18_params_fcn4.K, conv_18_params_fcn4.out_stride,
        (elem_t*)conv_17_out_fcn4, (elem_t*)conv_18_w_fcn4, (acc_t*)conv_18_b_fcn4, (elem_t*)conv_18_out_fcn4,
        RELU, conv_18_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_18_params_fcn4.I, conv_18_params_fcn4.J,
        conv_18_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_14_out_fcn4,
        (elem_t*)conv_18_out_fcn4,
        (elem_t*)conv_18_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_19_params_fcn4.I, conv_19_params_fcn4.J, conv_19_params_fcn4.K, conv_19_params_fcn4.out_stride,
        (elem_t*)conv_18_out_fcn4, (elem_t*)conv_19_w_fcn4, (acc_t*)conv_19_b_fcn4, (elem_t*)conv_19_out_fcn4,
        NO_ACTIVATION, conv_19_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_20
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_20_params_fcn4.batch_size, conv_20_params_fcn4.in_dim, conv_20_params_fcn4.in_channels,
        conv_20_params_fcn4.out_channels, conv_20_params_fcn4.out_dim,
        conv_20_params_fcn4.stride, conv_20_params_fcn4.dilation, conv_20_params_fcn4.padding, conv_20_params_fcn4.kernel_size,
        conv_20_params_fcn4.out_stride,

        (elem_t*)conv_19_out_fcn4, (elem_t*)conv_20_w_fcn4, (acc_t*)conv_20_b_fcn4, (elem_t*)conv_20_out_fcn4,

        NO_ACTIVATION, conv_20_params_fcn4.output_scale, 0,
        conv_20_params_fcn4.pool_size, 0, conv_20_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params_fcn4.I, conv_21_params_fcn4.J, conv_21_params_fcn4.K, conv_21_params_fcn4.out_stride,
        (elem_t*)conv_20_out_fcn4, (elem_t*)conv_21_w_fcn4, (acc_t*)conv_21_b_fcn4, (elem_t*)conv_21_out_fcn4,
        RELU, conv_21_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_21_params_fcn4.I, conv_21_params_fcn4.J,
        conv_21_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_18_out_fcn4,
        (elem_t*)conv_21_out_fcn4,
        (elem_t*)conv_21_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_22
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_22_params_fcn4.I, conv_22_params_fcn4.J, conv_22_params_fcn4.K, conv_22_params_fcn4.out_stride,
        (elem_t*)conv_21_out_fcn4, (elem_t*)conv_22_w_fcn4, (acc_t*)conv_22_b_fcn4, (elem_t*)conv_22_out_fcn4,
        NO_ACTIVATION, conv_22_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_23
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_23_params_fcn4.batch_size, conv_23_params_fcn4.in_dim, conv_23_params_fcn4.in_channels,
        conv_23_params_fcn4.out_channels, conv_23_params_fcn4.out_dim,
        conv_23_params_fcn4.stride, conv_23_params_fcn4.dilation, conv_23_params_fcn4.padding, conv_23_params_fcn4.kernel_size,
        conv_23_params_fcn4.out_stride,

        (elem_t*)conv_22_out_fcn4, (elem_t*)conv_23_w_fcn4, (acc_t*)conv_23_b_fcn4, (elem_t*)conv_23_out_fcn4,

        NO_ACTIVATION, conv_23_params_fcn4.output_scale, 0,
        conv_23_params_fcn4.pool_size, 0, conv_23_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params_fcn4.I, conv_24_params_fcn4.J, conv_24_params_fcn4.K, conv_24_params_fcn4.out_stride,
        (elem_t*)conv_23_out_fcn4, (elem_t*)conv_24_w_fcn4, (acc_t*)conv_24_b_fcn4, (elem_t*)conv_24_out_fcn4,
        RELU, conv_24_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_24_params_fcn4.I, conv_24_params_fcn4.J,
        conv_24_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_21_out_fcn4,
        (elem_t*)conv_24_out_fcn4,
        (elem_t*)conv_24_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_25
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_25_params_fcn4.I, conv_25_params_fcn4.J, conv_25_params_fcn4.K, conv_25_params_fcn4.out_stride,
        (elem_t*)conv_24_out_fcn4, (elem_t*)conv_25_w_fcn4, (acc_t*)conv_25_b_fcn4, (elem_t*)conv_25_out_fcn4,
        NO_ACTIVATION, conv_25_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_26
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_26_params_fcn4.batch_size, conv_26_params_fcn4.in_dim, conv_26_params_fcn4.in_channels,
        conv_26_params_fcn4.out_channels, conv_26_params_fcn4.out_dim,
        conv_26_params_fcn4.stride, conv_26_params_fcn4.dilation, conv_26_params_fcn4.padding, conv_26_params_fcn4.kernel_size,
        conv_26_params_fcn4.out_stride,

        (elem_t*)conv_25_out_fcn4, (elem_t*)conv_26_w_fcn4, (acc_t*)conv_26_b_fcn4, (elem_t*)conv_26_out_fcn4,

        NO_ACTIVATION, conv_26_params_fcn4.output_scale, 0,
        conv_26_params_fcn4.pool_size, 0, conv_26_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_27
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_27_params_fcn4.I, conv_27_params_fcn4.J, conv_27_params_fcn4.K, conv_27_params_fcn4.out_stride,
        (elem_t*)conv_26_out_fcn4, (elem_t*)conv_27_w_fcn4, (acc_t*)conv_27_b_fcn4, (elem_t*)conv_27_out_fcn4,
        RELU, conv_27_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Downsampling conv_24_out_fcn4
    // conv_28
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_28_params_fcn4.I, conv_28_params_fcn4.J, conv_28_params_fcn4.K, conv_28_params_fcn4.out_stride,
        (elem_t*)conv_24_out_fcn4, (elem_t*)conv_28_w_fcn4, (acc_t*)conv_28_b_fcn4, (elem_t*)conv_28_out_fcn4,
        NO_ACTIVATION, conv_28_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_27_params_fcn4.I, conv_27_params_fcn4.J,
        conv_27_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_28_out_fcn4,
        (elem_t*)conv_27_out_fcn4,
        (elem_t*)conv_27_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_29
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_29_params_fcn4.I, conv_29_params_fcn4.J, conv_29_params_fcn4.K, conv_29_params_fcn4.out_stride,
        (elem_t*)conv_27_out_fcn4, (elem_t*)conv_29_w_fcn4, (acc_t*)conv_29_b_fcn4, (elem_t*)conv_29_out_fcn4,
        NO_ACTIVATION, conv_29_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_30
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_30_params_fcn4.batch_size, conv_30_params_fcn4.in_dim, conv_30_params_fcn4.in_channels,
        conv_30_params_fcn4.out_channels, conv_30_params_fcn4.out_dim,
        conv_30_params_fcn4.stride, conv_30_params_fcn4.dilation, conv_30_params_fcn4.padding, conv_30_params_fcn4.kernel_size,
        conv_30_params_fcn4.out_stride,

        (elem_t*)conv_29_out_fcn4, (elem_t*)conv_30_w_fcn4, (acc_t*)conv_30_b_fcn4, (elem_t*)conv_30_out_fcn4,

        NO_ACTIVATION, conv_30_params_fcn4.output_scale, 0,
        conv_30_params_fcn4.pool_size, 0, conv_30_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_31
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_31_params_fcn4.I, conv_31_params_fcn4.J, conv_31_params_fcn4.K, conv_31_params_fcn4.out_stride,
        (elem_t*)conv_30_out_fcn4, (elem_t*)conv_31_w_fcn4, (acc_t*)conv_31_b_fcn4, (elem_t*)conv_31_out_fcn4,
        RELU, conv_31_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_31_params_fcn4.I, conv_31_params_fcn4.J,
        conv_31_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_27_out_fcn4,
        (elem_t*)conv_31_out_fcn4,
        (elem_t*)conv_31_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_32
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_32_params_fcn4.I, conv_32_params_fcn4.J, conv_32_params_fcn4.K, conv_32_params_fcn4.out_stride,
        (elem_t*)conv_31_out_fcn4, (elem_t*)conv_32_w_fcn4, (acc_t*)conv_32_b_fcn4, (elem_t*)conv_32_out_fcn4,
        NO_ACTIVATION, conv_32_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_33
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_33_params_fcn4.batch_size, conv_33_params_fcn4.in_dim, conv_33_params_fcn4.in_channels,
        conv_33_params_fcn4.out_channels, conv_33_params_fcn4.out_dim,
        conv_33_params_fcn4.stride, conv_33_params_fcn4.dilation, conv_33_params_fcn4.padding, conv_33_params_fcn4.kernel_size,
        conv_33_params_fcn4.out_stride,

        (elem_t*)conv_32_out_fcn4, (elem_t*)conv_33_w_fcn4, (acc_t*)conv_33_b_fcn4, (elem_t*)conv_33_out_fcn4,

        NO_ACTIVATION, conv_33_params_fcn4.output_scale, 0,
        conv_33_params_fcn4.pool_size, 0, conv_33_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_34
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_34_params_fcn4.I, conv_34_params_fcn4.J, conv_34_params_fcn4.K, conv_34_params_fcn4.out_stride,
        (elem_t*)conv_33_out_fcn4, (elem_t*)conv_34_w_fcn4, (acc_t*)conv_34_b_fcn4, (elem_t*)conv_34_out_fcn4,
        RELU, conv_34_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_34_params_fcn4.I, conv_34_params_fcn4.J,
        conv_34_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_31_out_fcn4,
        (elem_t*)conv_34_out_fcn4,
        (elem_t*)conv_34_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_35
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_35_params_fcn4.I, conv_35_params_fcn4.J, conv_35_params_fcn4.K, conv_35_params_fcn4.out_stride,
        (elem_t*)conv_34_out_fcn4, (elem_t*)conv_35_w_fcn4, (acc_t*)conv_35_b_fcn4, (elem_t*)conv_35_out_fcn4,
        NO_ACTIVATION, conv_35_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_36
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_36_params_fcn4.batch_size, conv_36_params_fcn4.in_dim, conv_36_params_fcn4.in_channels,
        conv_36_params_fcn4.out_channels, conv_36_params_fcn4.out_dim,
        conv_36_params_fcn4.stride, conv_36_params_fcn4.dilation, conv_36_params_fcn4.padding, conv_36_params_fcn4.kernel_size,
        conv_36_params_fcn4.out_stride,

        (elem_t*)conv_35_out_fcn4, (elem_t*)conv_36_w_fcn4, (acc_t*)conv_36_b_fcn4, (elem_t*)conv_36_out_fcn4,

        NO_ACTIVATION, conv_36_params_fcn4.output_scale, 0,
        conv_36_params_fcn4.pool_size, 0, conv_36_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_37
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_37_params_fcn4.I, conv_37_params_fcn4.J, conv_37_params_fcn4.K, conv_37_params_fcn4.out_stride,
        (elem_t*)conv_36_out_fcn4, (elem_t*)conv_37_w_fcn4, (acc_t*)conv_37_b_fcn4, (elem_t*)conv_37_out_fcn4,
        RELU, conv_37_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_37_params_fcn4.I, conv_37_params_fcn4.J,
        conv_37_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_34_out_fcn4,
        (elem_t*)conv_37_out_fcn4,
        (elem_t*)conv_37_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_38
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_38_params_fcn4.I, conv_38_params_fcn4.J, conv_38_params_fcn4.K, conv_38_params_fcn4.out_stride,
        (elem_t*)conv_37_out_fcn4, (elem_t*)conv_38_w_fcn4, (acc_t*)conv_38_b_fcn4, (elem_t*)conv_38_out_fcn4,
        NO_ACTIVATION, conv_38_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_39
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_39_params_fcn4.batch_size, conv_39_params_fcn4.in_dim, conv_39_params_fcn4.in_channels,
        conv_39_params_fcn4.out_channels, conv_39_params_fcn4.out_dim,
        conv_39_params_fcn4.stride, conv_39_params_fcn4.dilation, conv_39_params_fcn4.padding, conv_39_params_fcn4.kernel_size,
        conv_39_params_fcn4.out_stride,

        (elem_t*)conv_38_out_fcn4, (elem_t*)conv_39_w_fcn4, (acc_t*)conv_39_b_fcn4, (elem_t*)conv_39_out_fcn4,

        NO_ACTIVATION, conv_39_params_fcn4.output_scale, 0,
        conv_39_params_fcn4.pool_size, 0, conv_39_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_40
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_40_params_fcn4.I, conv_40_params_fcn4.J, conv_40_params_fcn4.K, conv_40_params_fcn4.out_stride,
        (elem_t*)conv_39_out_fcn4, (elem_t*)conv_40_w_fcn4, (acc_t*)conv_40_b_fcn4, (elem_t*)conv_40_out_fcn4,
        RELU, conv_40_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_40_params_fcn4.I, conv_40_params_fcn4.J,
        conv_40_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_37_out_fcn4,
        (elem_t*)conv_40_out_fcn4,
        (elem_t*)conv_40_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_41
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_41_params_fcn4.I, conv_41_params_fcn4.J, conv_41_params_fcn4.K, conv_41_params_fcn4.out_stride,
        (elem_t*)conv_40_out_fcn4, (elem_t*)conv_41_w_fcn4, (acc_t*)conv_41_b_fcn4, (elem_t*)conv_41_out_fcn4,
        NO_ACTIVATION, conv_41_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_42
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_42_params_fcn4.batch_size, conv_42_params_fcn4.in_dim, conv_42_params_fcn4.in_channels,
        conv_42_params_fcn4.out_channels, conv_42_params_fcn4.out_dim,
        conv_42_params_fcn4.stride, conv_42_params_fcn4.dilation, conv_42_params_fcn4.padding, conv_42_params_fcn4.kernel_size,
        conv_42_params_fcn4.out_stride,

        (elem_t*)conv_41_out_fcn4, (elem_t*)conv_42_w_fcn4, (acc_t*)conv_42_b_fcn4, (elem_t*)conv_42_out_fcn4,

        NO_ACTIVATION, conv_42_params_fcn4.output_scale, 0,
        conv_42_params_fcn4.pool_size, 0, conv_42_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_43
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_43_params_fcn4.I, conv_43_params_fcn4.J, conv_43_params_fcn4.K, conv_43_params_fcn4.out_stride,
        (elem_t*)conv_42_out_fcn4, (elem_t*)conv_43_w_fcn4, (acc_t*)conv_43_b_fcn4, (elem_t*)conv_43_out_fcn4,
        RELU, conv_43_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_43_params_fcn4.I, conv_43_params_fcn4.J,
        conv_43_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_40_out_fcn4,
        (elem_t*)conv_43_out_fcn4,
        (elem_t*)conv_43_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_44
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_44_params_fcn4.I, conv_44_params_fcn4.J, conv_44_params_fcn4.K, conv_44_params_fcn4.out_stride,
        (elem_t*)conv_43_out_fcn4, (elem_t*)conv_44_w_fcn4, (acc_t*)conv_44_b_fcn4, (elem_t*)conv_44_out_fcn4,
        NO_ACTIVATION, conv_44_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_45
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_45_params_fcn4.batch_size, conv_45_params_fcn4.in_dim, conv_45_params_fcn4.in_channels,
        conv_45_params_fcn4.out_channels, conv_45_params_fcn4.out_dim,
        conv_45_params_fcn4.stride, conv_45_params_fcn4.dilation, conv_45_params_fcn4.padding, conv_45_params_fcn4.kernel_size,
        conv_45_params_fcn4.out_stride,

        (elem_t*)conv_44_out_fcn4, (elem_t*)conv_45_w_fcn4, (acc_t*)conv_45_b_fcn4, (elem_t*)conv_45_out_fcn4,

        NO_ACTIVATION, conv_45_params_fcn4.output_scale, 0,
        conv_45_params_fcn4.pool_size, 0, conv_45_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_46
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_46_params_fcn4.I, conv_46_params_fcn4.J, conv_46_params_fcn4.K, conv_46_params_fcn4.out_stride,
        (elem_t*)conv_45_out_fcn4, (elem_t*)conv_46_w_fcn4, (acc_t*)conv_46_b_fcn4, (elem_t*)conv_46_out_fcn4,
        RELU, conv_46_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Downsampling conv_43_out_fcn4
    // conv_47
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_47_params_fcn4.I, conv_47_params_fcn4.J, conv_47_params_fcn4.K, conv_47_params_fcn4.out_stride,
        (elem_t*)conv_43_out_fcn4, (elem_t*)conv_47_w_fcn4, (acc_t*)conv_47_b_fcn4, (elem_t*)conv_47_out_fcn4,
        NO_ACTIVATION, conv_47_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_46_params_fcn4.I, conv_46_params_fcn4.J,
        conv_46_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_47_out_fcn4,
        (elem_t*)conv_46_out_fcn4,
        (elem_t*)conv_46_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_48
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_48_params_fcn4.I, conv_48_params_fcn4.J, conv_48_params_fcn4.K, conv_48_params_fcn4.out_stride,
        (elem_t*)conv_46_out_fcn4, (elem_t*)conv_48_w_fcn4, (acc_t*)conv_48_b_fcn4, (elem_t*)conv_48_out_fcn4,
        NO_ACTIVATION, conv_48_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_49
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_49_params_fcn4.batch_size, conv_49_params_fcn4.in_dim, conv_49_params_fcn4.in_channels,
        conv_49_params_fcn4.out_channels, conv_49_params_fcn4.out_dim,
        conv_49_params_fcn4.stride, conv_49_params_fcn4.dilation, conv_49_params_fcn4.padding, conv_49_params_fcn4.kernel_size,
        conv_49_params_fcn4.out_stride,

        (elem_t*)conv_48_out_fcn4, (elem_t*)conv_49_w_fcn4, (acc_t*)conv_49_b_fcn4, (elem_t*)conv_49_out_fcn4,

        NO_ACTIVATION, conv_49_params_fcn4.output_scale, 0,
        conv_49_params_fcn4.pool_size, 0, conv_49_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_50
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_50_params_fcn4.I, conv_50_params_fcn4.J, conv_50_params_fcn4.K, conv_50_params_fcn4.out_stride,
        (elem_t*)conv_49_out_fcn4, (elem_t*)conv_50_w_fcn4, (acc_t*)conv_50_b_fcn4, (elem_t*)conv_50_out_fcn4,
        RELU, conv_50_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_50_params_fcn4.I, conv_50_params_fcn4.J,
        conv_50_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_46_out_fcn4,
        (elem_t*)conv_50_out_fcn4,
        (elem_t*)conv_50_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_51
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_51_params_fcn4.I, conv_51_params_fcn4.J, conv_51_params_fcn4.K, conv_51_params_fcn4.out_stride,
        (elem_t*)conv_50_out_fcn4, (elem_t*)conv_51_w_fcn4, (acc_t*)conv_51_b_fcn4, (elem_t*)conv_51_out_fcn4,
        NO_ACTIVATION, conv_51_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_52
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_52_params_fcn4.batch_size, conv_52_params_fcn4.in_dim, conv_52_params_fcn4.in_channels,
        conv_52_params_fcn4.out_channels, conv_52_params_fcn4.out_dim,
        conv_52_params_fcn4.stride, conv_52_params_fcn4.dilation, conv_52_params_fcn4.padding, conv_52_params_fcn4.kernel_size,
        conv_52_params_fcn4.out_stride,

        (elem_t*)conv_51_out_fcn4, (elem_t*)conv_52_w_fcn4, (acc_t*)conv_52_b_fcn4, (elem_t*)conv_52_out_fcn4,

        NO_ACTIVATION, conv_52_params_fcn4.output_scale, 0,
        conv_52_params_fcn4.pool_size, 0, conv_52_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_53
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_53_params_fcn4.I, conv_53_params_fcn4.J, conv_53_params_fcn4.K, conv_53_params_fcn4.out_stride,
        (elem_t*)conv_52_out_fcn4, (elem_t*)conv_53_w_fcn4, (acc_t*)conv_53_b_fcn4, (elem_t*)conv_53_out_fcn4,
        RELU, conv_53_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_cid(conv_53_params_fcn4.I, conv_53_params_fcn4.J,
        conv_53_params_fcn4.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_50_out_fcn4,
        (elem_t*)conv_53_out_fcn4,
        (elem_t*)conv_53_out_fcn4,
        false,
         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
        
    // conv_54
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_54_params_fcn4.batch_size, conv_54_params_fcn4.in_dim, conv_54_params_fcn4.in_channels,
        conv_54_params_fcn4.out_channels, conv_54_params_fcn4.out_dim,
        conv_54_params_fcn4.stride, conv_54_params_fcn4.dilation, conv_54_params_fcn4.padding, conv_54_params_fcn4.kernel_size,
        conv_54_params_fcn4.out_stride,

        (elem_t*)conv_53_out_fcn4, (elem_t*)conv_54_w_fcn4, (acc_t*)conv_54_b_fcn4, (elem_t*)conv_54_out_fcn4,

        RELU, conv_54_params_fcn4.output_scale, 0,
        conv_54_params_fcn4.pool_size, 0, conv_54_params_fcn4.pool_padding, false,

         WS, orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
        
    // conv_55
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_55_params_fcn4.I, conv_55_params_fcn4.J, conv_55_params_fcn4.K, conv_55_params_fcn4.out_stride,
        (elem_t*)conv_54_out_fcn4, (elem_t*)conv_55_w_fcn4, (acc_t*)conv_55_b_fcn4, (elem_t*)conv_55_out_fcn4,
        NO_ACTIVATION, conv_55_params_fcn4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[54] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif
/*
// interpolation

    start = read_cycles();
    tiled_interpolate_auto(conv_55_params_fcn4.in_dim, conv_55_params_fcn4.out_stride, 224,
	(elem_t*) conv_55_out_fcn4, (elem_t*) image_out_fcn4, OROW_DIVIDE, cid);
    end = read_cycles();
    other_cycles = end - start;
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_fcnnet);
#endif        
 */

    for(int i = 0; i < num_cycle; i++){
      if(i < 55){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 71){
        cycles[cid][i] = resadd_cycles[i - 55];
      }
      else{
        if(i == 71) cycles[cid][i] = total_conv_cycles;
        if(i == 72) cycles[cid][i] = total_resadd_cycles;
        if(i == 73) cycles[cid][i] = total_conv_cycles + total_resadd_cycles + other_cycles;
      }
    }

    return cycles[cid];
#undef num_cycle
}
