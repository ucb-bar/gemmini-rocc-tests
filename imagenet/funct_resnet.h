
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet_orow_params.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* resnet_function(int cid, int num_cycle, uint64_t cycles[num_cycle], int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_res){
#else
uint64_t* resnet_function(int cid, int num_cycle, uint64_t cycles[num_cycle], int orow_divide, int batch_divide, int target_util){
#endif

    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[20];
    uint64_t matmul_cycles[34];
    uint64_t res_add_cycles[16];
printf("entered resnet_function\n");
    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
printf("barrier wait work \n");

    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params.batch_size, conv_1_params.in_dim, conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_dim,
        conv_1_params.stride, 1, conv_1_params.padding, conv_1_params.kernel_size,
        conv_1_params.out_stride,

        (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

        RELU, conv_1_params.output_scale, 0,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;

printf("after layer 1 - cid: %d\n", cid);   
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif             
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params.I, conv_2_params.J, conv_2_params.K, conv_2_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,
        RELU, conv_2_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] = end - start;

printf("after layer 2 - cid: %d\n", cid);   
   #if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_dim,
        conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,
        conv_3_params.out_stride,

        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

        RELU, conv_3_params.output_scale, 0,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;

printf("after layer 3 - cid: %d\n", cid);   

#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_4
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_4_params.I, conv_4_params.J, conv_4_params.K, conv_4_params.out_stride,
        (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,
        NO_ACTIVATION, conv_4_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // Downsampling conv_1_out_pooled
    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params.I, conv_5_params.J, conv_5_params.K, conv_5_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,
        NO_ACTIVATION, conv_5_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_6
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_6_params.I, conv_6_params.J, conv_6_params.K, conv_6_params.out_stride,
        (elem_t*)conv_4_out, (elem_t*)conv_6_w, (acc_t*)conv_6_b, (elem_t*)conv_6_out,
        RELU, conv_6_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_7_params.batch_size, conv_7_params.in_dim, conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_dim,
        conv_7_params.stride, 1, conv_7_params.padding, conv_7_params.kernel_size,
        conv_7_params.out_stride,

        (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

        RELU, conv_7_params.output_scale, 0,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params.I, conv_8_params.J, conv_8_params.K, conv_8_params.out_stride,
        (elem_t*)conv_7_out, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,
        NO_ACTIVATION, conv_8_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params.I, conv_9_params.J, conv_9_params.K, conv_9_params.out_stride,
        (elem_t*)conv_8_out, (elem_t*)conv_9_w, (acc_t*)conv_9_b, (elem_t*)conv_9_out,
        RELU, conv_9_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_10
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_10_params.batch_size, conv_10_params.in_dim, conv_10_params.in_channels,
        conv_10_params.out_channels, conv_10_params.out_dim,
        conv_10_params.stride, 1, conv_10_params.padding, conv_10_params.kernel_size,
        conv_10_params.out_stride,

        (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

        RELU, conv_10_params.output_scale, 0,
        conv_10_params.pool_size, 0, conv_10_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params.I, conv_11_params.J, conv_11_params.K, conv_11_params.out_stride,
        (elem_t*)conv_10_out, (elem_t*)conv_11_w, (acc_t*)conv_11_b, (elem_t*)conv_11_out,
        NO_ACTIVATION, conv_11_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_12
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_12_params.I, conv_12_params.J, conv_12_params.K, conv_12_params.out_stride,
        (elem_t*)conv_11_out, (elem_t*)conv_12_w, (acc_t*)conv_12_b, (elem_t*)conv_12_out,
        RELU, conv_12_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_13
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_13_params.batch_size, conv_13_params.in_dim, conv_13_params.in_channels,
        conv_13_params.out_channels, conv_13_params.out_dim,
        conv_13_params.stride, 1, conv_13_params.padding, conv_13_params.kernel_size,
        conv_13_params.out_stride,

        (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

        RELU, conv_13_params.output_scale, 0,
        conv_13_params.pool_size, 0, conv_13_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params.I, conv_14_params.J, conv_14_params.K, conv_14_params.out_stride,
        (elem_t*)conv_13_out, (elem_t*)conv_14_w, (acc_t*)conv_14_b, (elem_t*)conv_14_out,
        NO_ACTIVATION, conv_14_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // Downsampling conv_11_out
    // conv_15
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_15_params.batch_size, conv_15_params.in_dim, conv_15_params.in_channels,
        conv_15_params.out_channels, conv_15_params.out_dim,
        conv_15_params.stride, 1, conv_15_params.padding, conv_15_params.kernel_size,
        conv_15_params.out_stride,

        (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

        NO_ACTIVATION, conv_15_params.output_scale, 0,
        conv_15_params.pool_size, 0, conv_15_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_16
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_16_params.I, conv_16_params.J, conv_16_params.K, conv_16_params.out_stride,
        (elem_t*)conv_14_out, (elem_t*)conv_16_w, (acc_t*)conv_16_b, (elem_t*)conv_16_out,
        RELU, conv_16_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_17_params.batch_size, conv_17_params.in_dim, conv_17_params.in_channels,
        conv_17_params.out_channels, conv_17_params.out_dim,
        conv_17_params.stride, 1, conv_17_params.padding, conv_17_params.kernel_size,
        conv_17_params.out_stride,

        (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

        RELU, conv_17_params.output_scale, 0,
        conv_17_params.pool_size, 0, conv_17_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params.I, conv_18_params.J, conv_18_params.K, conv_18_params.out_stride,
        (elem_t*)conv_17_out, (elem_t*)conv_18_w, (acc_t*)conv_18_b, (elem_t*)conv_18_out,
        NO_ACTIVATION, conv_18_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_19_params.I, conv_19_params.J, conv_19_params.K, conv_19_params.out_stride,
        (elem_t*)conv_18_out, (elem_t*)conv_19_w, (acc_t*)conv_19_b, (elem_t*)conv_19_out,
        RELU, conv_19_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_20
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_20_params.batch_size, conv_20_params.in_dim, conv_20_params.in_channels,
        conv_20_params.out_channels, conv_20_params.out_dim,
        conv_20_params.stride, 1, conv_20_params.padding, conv_20_params.kernel_size,
        conv_20_params.out_stride,

        (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

        RELU, conv_20_params.output_scale, 0,
        conv_20_params.pool_size, 0, conv_20_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params.I, conv_21_params.J, conv_21_params.K, conv_21_params.out_stride,
        (elem_t*)conv_20_out, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)conv_21_out,
        NO_ACTIVATION, conv_21_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_22
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_22_params.I, conv_22_params.J, conv_22_params.K, conv_22_params.out_stride,
        (elem_t*)conv_21_out, (elem_t*)conv_22_w, (acc_t*)conv_22_b, (elem_t*)conv_22_out,
        RELU, conv_22_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_23
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_23_params.batch_size, conv_23_params.in_dim, conv_23_params.in_channels,
        conv_23_params.out_channels, conv_23_params.out_dim,
        conv_23_params.stride, 1, conv_23_params.padding, conv_23_params.kernel_size,
        conv_23_params.out_stride,

        (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

        RELU, conv_23_params.output_scale, 0,
        conv_23_params.pool_size, 0, conv_23_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params.I, conv_24_params.J, conv_24_params.K, conv_24_params.out_stride,
        (elem_t*)conv_23_out, (elem_t*)conv_24_w, (acc_t*)conv_24_b, (elem_t*)conv_24_out,
        NO_ACTIVATION, conv_24_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_25
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_25_params.I, conv_25_params.J, conv_25_params.K, conv_25_params.out_stride,
        (elem_t*)conv_24_out, (elem_t*)conv_25_w, (acc_t*)conv_25_b, (elem_t*)conv_25_out,
        RELU, conv_25_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_26
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_26_params.batch_size, conv_26_params.in_dim, conv_26_params.in_channels,
        conv_26_params.out_channels, conv_26_params.out_dim,
        conv_26_params.stride, 1, conv_26_params.padding, conv_26_params.kernel_size,
        conv_26_params.out_stride,

        (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

        RELU, conv_26_params.output_scale, 0,
        conv_26_params.pool_size, 0, conv_26_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_27
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_27_params.I, conv_27_params.J, conv_27_params.K, conv_27_params.out_stride,
        (elem_t*)conv_26_out, (elem_t*)conv_27_w, (acc_t*)conv_27_b, (elem_t*)conv_27_out,
        NO_ACTIVATION, conv_27_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // Downsampling conv_24_out
    // conv_28
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_28_params.batch_size, conv_28_params.in_dim, conv_28_params.in_channels,
        conv_28_params.out_channels, conv_28_params.out_dim,
        conv_28_params.stride, 1, conv_28_params.padding, conv_28_params.kernel_size,
        conv_28_params.out_stride,

        (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,

        NO_ACTIVATION, conv_28_params.output_scale, 0,
        conv_28_params.pool_size, 0, conv_28_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_29
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_29_params.I, conv_29_params.J, conv_29_params.K, conv_29_params.out_stride,
        (elem_t*)conv_27_out, (elem_t*)conv_29_w, (acc_t*)conv_29_b, (elem_t*)conv_29_out,
        RELU, conv_29_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_30
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_30_params.batch_size, conv_30_params.in_dim, conv_30_params.in_channels,
        conv_30_params.out_channels, conv_30_params.out_dim,
        conv_30_params.stride, 1, conv_30_params.padding, conv_30_params.kernel_size,
        conv_30_params.out_stride,

        (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

        RELU, conv_30_params.output_scale, 0,
        conv_30_params.pool_size, 0, conv_30_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_31
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_31_params.I, conv_31_params.J, conv_31_params.K, conv_31_params.out_stride,
        (elem_t*)conv_30_out, (elem_t*)conv_31_w, (acc_t*)conv_31_b, (elem_t*)conv_31_out,
        NO_ACTIVATION, conv_31_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_32
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_32_params.I, conv_32_params.J, conv_32_params.K, conv_32_params.out_stride,
        (elem_t*)conv_31_out, (elem_t*)conv_32_w, (acc_t*)conv_32_b, (elem_t*)conv_32_out,
        RELU, conv_32_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_33
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_33_params.batch_size, conv_33_params.in_dim, conv_33_params.in_channels,
        conv_33_params.out_channels, conv_33_params.out_dim,
        conv_33_params.stride, 1, conv_33_params.padding, conv_33_params.kernel_size,
        conv_33_params.out_stride,

        (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

        RELU, conv_33_params.output_scale, 0,
        conv_33_params.pool_size, 0, conv_33_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_34
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_34_params.I, conv_34_params.J, conv_34_params.K, conv_34_params.out_stride,
        (elem_t*)conv_33_out, (elem_t*)conv_34_w, (acc_t*)conv_34_b, (elem_t*)conv_34_out,
        NO_ACTIVATION, conv_34_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_35
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_35_params.I, conv_35_params.J, conv_35_params.K, conv_35_params.out_stride,
        (elem_t*)conv_34_out, (elem_t*)conv_35_w, (acc_t*)conv_35_b, (elem_t*)conv_35_out,
        RELU, conv_35_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_36
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_36_params.batch_size, conv_36_params.in_dim, conv_36_params.in_channels,
        conv_36_params.out_channels, conv_36_params.out_dim,
        conv_36_params.stride, 1, conv_36_params.padding, conv_36_params.kernel_size,
        conv_36_params.out_stride,

        (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

        RELU, conv_36_params.output_scale, 0,
        conv_36_params.pool_size, 0, conv_36_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_37
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_37_params.I, conv_37_params.J, conv_37_params.K, conv_37_params.out_stride,
        (elem_t*)conv_36_out, (elem_t*)conv_37_w, (acc_t*)conv_37_b, (elem_t*)conv_37_out,
        NO_ACTIVATION, conv_37_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_38
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_38_params.I, conv_38_params.J, conv_38_params.K, conv_38_params.out_stride,
        (elem_t*)conv_37_out, (elem_t*)conv_38_w, (acc_t*)conv_38_b, (elem_t*)conv_38_out,
        RELU, conv_38_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_39
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_39_params.batch_size, conv_39_params.in_dim, conv_39_params.in_channels,
        conv_39_params.out_channels, conv_39_params.out_dim,
        conv_39_params.stride, 1, conv_39_params.padding, conv_39_params.kernel_size,
        conv_39_params.out_stride,

        (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

        RELU, conv_39_params.output_scale, 0,
        conv_39_params.pool_size, 0, conv_39_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_40
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_40_params.I, conv_40_params.J, conv_40_params.K, conv_40_params.out_stride,
        (elem_t*)conv_39_out, (elem_t*)conv_40_w, (acc_t*)conv_40_b, (elem_t*)conv_40_out,
        NO_ACTIVATION, conv_40_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_41
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_41_params.I, conv_41_params.J, conv_41_params.K, conv_41_params.out_stride,
        (elem_t*)conv_40_out, (elem_t*)conv_41_w, (acc_t*)conv_41_b, (elem_t*)conv_41_out,
        RELU, conv_41_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_42
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_42_params.batch_size, conv_42_params.in_dim, conv_42_params.in_channels,
        conv_42_params.out_channels, conv_42_params.out_dim,
        conv_42_params.stride, 1, conv_42_params.padding, conv_42_params.kernel_size,
        conv_42_params.out_stride,

        (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

        RELU, conv_42_params.output_scale, 0,
        conv_42_params.pool_size, 0, conv_42_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_43
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_43_params.I, conv_43_params.J, conv_43_params.K, conv_43_params.out_stride,
        (elem_t*)conv_42_out, (elem_t*)conv_43_w, (acc_t*)conv_43_b, (elem_t*)conv_43_out,
        NO_ACTIVATION, conv_43_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[26] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_44
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_44_params.I, conv_44_params.J, conv_44_params.K, conv_44_params.out_stride,
        (elem_t*)conv_43_out, (elem_t*)conv_44_w, (acc_t*)conv_44_b, (elem_t*)conv_44_out,
        RELU, conv_44_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_45
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_45_params.batch_size, conv_45_params.in_dim, conv_45_params.in_channels,
        conv_45_params.out_channels, conv_45_params.out_dim,
        conv_45_params.stride, 1, conv_45_params.padding, conv_45_params.kernel_size,
        conv_45_params.out_stride,

        (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

        RELU, conv_45_params.output_scale, 0,
        conv_45_params.pool_size, 0, conv_45_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_46
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_46_params.I, conv_46_params.J, conv_46_params.K, conv_46_params.out_stride,
        (elem_t*)conv_45_out, (elem_t*)conv_46_w, (acc_t*)conv_46_b, (elem_t*)conv_46_out,
        NO_ACTIVATION, conv_46_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // Downsampling conv_43_out
    // conv_47
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_47_params.batch_size, conv_47_params.in_dim, conv_47_params.in_channels,
        conv_47_params.out_channels, conv_47_params.out_dim,
        conv_47_params.stride, 1, conv_47_params.padding, conv_47_params.kernel_size,
        conv_47_params.out_stride,

        (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,

        NO_ACTIVATION, conv_47_params.output_scale, 0,
        conv_47_params.pool_size, 0, conv_47_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_48
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_48_params.I, conv_48_params.J, conv_48_params.K, conv_48_params.out_stride,
        (elem_t*)conv_46_out, (elem_t*)conv_48_w, (acc_t*)conv_48_b, (elem_t*)conv_48_out,
        RELU, conv_48_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_49
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_49_params.batch_size, conv_49_params.in_dim, conv_49_params.in_channels,
        conv_49_params.out_channels, conv_49_params.out_dim,
        conv_49_params.stride, 1, conv_49_params.padding, conv_49_params.kernel_size,
        conv_49_params.out_stride,

        (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

        RELU, conv_49_params.output_scale, 0,
        conv_49_params.pool_size, 0, conv_49_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_50
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_50_params.I, conv_50_params.J, conv_50_params.K, conv_50_params.out_stride,
        (elem_t*)conv_49_out, (elem_t*)conv_50_w, (acc_t*)conv_50_b, (elem_t*)conv_50_out,
        NO_ACTIVATION, conv_50_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[30] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_51
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_51_params.I, conv_51_params.J, conv_51_params.K, conv_51_params.out_stride,
        (elem_t*)conv_50_out, (elem_t*)conv_51_w, (acc_t*)conv_51_b, (elem_t*)conv_51_out,
        RELU, conv_51_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[31] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // conv_52
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_52_params.batch_size, conv_52_params.in_dim, conv_52_params.in_channels,
        conv_52_params.out_channels, conv_52_params.out_dim,
        conv_52_params.stride, 1, conv_52_params.padding, conv_52_params.kernel_size,
        conv_52_params.out_stride,

        (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

        RELU, conv_52_params.output_scale, 0,
        conv_52_params.pool_size, 0, conv_52_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif        
        
    // conv_53
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_53_params.I, conv_53_params.J, conv_53_params.K, conv_53_params.out_stride,
        (elem_t*)conv_52_out, (elem_t*)conv_53_w, (acc_t*)conv_53_b, (elem_t*)conv_53_out,
        NO_ACTIVATION, conv_53_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[32] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
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
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    res_add_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif
        
    // Global averaging
    
    static elem_t average[1][2048] row_align(MAX_BLOCK_LEN);

    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_53_out, average, conv_53_params.batch_size,                         
            conv_53_params.out_channels, conv_53_params.out_dim, WS);
       

    end = read_cycles();
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif

    // fc_54
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_54_params.I, fc_54_params.J, fc_54_params.K, fc_54_params.out_stride,
        (elem_t*)average, (elem_t*)fc_54_w, (acc_t*)fc_54_b, (elem_t*)fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[33] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier_res);
#endif   

    for(int i = 0; i < num_cycle; i++){
      if(i < 20){
        cycles[i] = conv_cycles[i];
      }
      else if(i < 54){
        cycles[i] = matmul_cycles[i - 20];
      }
      else if (i < 70){
        cycles[i] = res_add_cycles[i - 54];
      }
      else{
        if(i == 70) cycles[i] = total_conv_cycles;
        if(i == 71) cycles[i] = total_matmul_cycles;
        if(i == 72) cycles[i] = total_resadd_cycles;
        if(i == 73) cycles[i] = total_conv_cycles + total_matmul_cycles + total_resadd_cycles + other_cycles;
      }
    }

}
