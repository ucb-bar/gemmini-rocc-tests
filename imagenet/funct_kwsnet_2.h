// resnet26 for keyword spotting

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "kwsnet_params_2.h"
#include "fingerprints.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif


#ifndef BAREMETAL
uint64_t* kwsnet_function_2(int cid, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_kws2){
#else
uint64_t* kwsnet_function_2(int cid, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (25+1+11+4)

  static uint64_t cycles[num_proc][num_cycle];
    uint64_t start, end;
    uint64_t total_pool_cycles = 0, total_conv_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[25];
    uint64_t pool_cycles[1];
    uint64_t resadd_cycles[11];
   //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif

    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_1_params_kws2.batch_size, conv_1_params_kws2.in_dim, conv_1_params_kws2.in_channels,
        conv_1_params_kws2.out_channels, conv_1_params_kws2.out_dim,
        conv_1_params_kws2.stride, 1, conv_1_params_kws2.padding, conv_1_params_kws2.kernel_size,
        conv_1_params_kws2.out_stride, conv_1_params_kws2.in_channels, 64,

        (elem_t*)fingerprints_2, (elem_t*)conv_1_w_kws2, (acc_t*)conv_1_b_kws2, (elem_t*)conv_1_out_kws2,

        RELU, conv_1_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_1_params_kws2.pool_size, conv_1_params_kws2.pool_stride, conv_1_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_kws2.batch_size,
        conv_1_params_kws2.out_channels, conv_1_params_kws2.out_dim, conv_1_params_kws2.out_dim_pooled,
        conv_1_params_kws2.out_stride,
        conv_1_params_kws2.pool_size, conv_1_params_kws2.pool_stride, conv_1_params_kws2.pool_padding,

        (elem_t*)conv_1_out_kws2, (elem_t*)conv_1_out_kws2_pooled,
    orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
        
    // conv_2
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_2_params_kws2.batch_size, conv_2_params_kws2.in_dim, conv_2_params_kws2.in_channels,
        conv_2_params_kws2.out_channels, conv_2_params_kws2.out_dim,
        conv_2_params_kws2.stride, 1, conv_2_params_kws2.padding, conv_2_params_kws2.kernel_size,
        conv_2_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_1_out_kws2_pooled, (elem_t*)conv_2_w_kws2, (acc_t*)conv_2_b_kws2, (elem_t*)conv_2_out_kws2,

        RELU, conv_2_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_2_params_kws2.pool_size, conv_2_params_kws2.pool_stride, conv_2_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_3_params_kws2.batch_size, conv_3_params_kws2.in_dim, conv_3_params_kws2.in_channels,
        conv_3_params_kws2.out_channels, conv_3_params_kws2.out_dim,
        conv_3_params_kws2.stride, 1, conv_3_params_kws2.padding, conv_3_params_kws2.kernel_size,
        conv_3_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_2_out_kws2, (elem_t*)conv_3_w_kws2, (acc_t*)conv_3_b_kws2, (elem_t*)conv_3_out_kws2,

        RELU, conv_3_params_kws2.output_scale, 0,
        conv_3_params_kws2.pool_size, 0, conv_3_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_3_params_kws2.I, conv_3_params_kws2.J,
        conv_3_params_kws2.res_scale,
        conv_3_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_1_out_kws2_pooled,
        (elem_t*)conv_3_out_kws2,
        (elem_t*)conv_3_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_4
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_4_params_kws2.batch_size, conv_4_params_kws2.in_dim, conv_4_params_kws2.in_channels,
        conv_4_params_kws2.out_channels, conv_4_params_kws2.out_dim,
        conv_4_params_kws2.stride, 1, conv_4_params_kws2.padding, conv_4_params_kws2.kernel_size,
        conv_4_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_3_out_kws2, (elem_t*)conv_4_w_kws2, (acc_t*)conv_4_b_kws2, (elem_t*)conv_4_out_kws2,

        RELU, conv_4_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_4_params_kws2.pool_size, conv_4_params_kws2.pool_stride, conv_4_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_5
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_5_params_kws2.batch_size, conv_5_params_kws2.in_dim, conv_5_params_kws2.in_channels,
        conv_5_params_kws2.out_channels, conv_5_params_kws2.out_dim,
        conv_5_params_kws2.stride, 1, conv_5_params_kws2.padding, conv_5_params_kws2.kernel_size,
        conv_5_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_4_out_kws2, (elem_t*)conv_5_w_kws2, (acc_t*)conv_5_b_kws2, (elem_t*)conv_5_out_kws2,

        RELU, conv_5_params_kws2.output_scale, 0,
        conv_5_params_kws2.pool_size, 0, conv_5_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_5_params_kws2.I, conv_5_params_kws2.J,
        conv_5_params_kws2.res_scale,
        conv_5_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_3_out_kws2,
        (elem_t*)conv_5_out_kws2,
        (elem_t*)conv_5_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
    
    // conv_6
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_6_params_kws2.batch_size, conv_6_params_kws2.in_dim, conv_6_params_kws2.in_channels,
        conv_6_params_kws2.out_channels, conv_6_params_kws2.out_dim,
        conv_6_params_kws2.stride, 1, conv_6_params_kws2.padding, conv_6_params_kws2.kernel_size,
        conv_6_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_5_out_kws2, (elem_t*)conv_6_w_kws2, (acc_t*)conv_6_b_kws2, (elem_t*)conv_6_out_kws2,

        RELU, conv_6_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_6_params_kws2.pool_size, conv_6_params_kws2.pool_stride, conv_6_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_7
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_7_params_kws2.batch_size, conv_7_params_kws2.in_dim, conv_7_params_kws2.in_channels,
        conv_7_params_kws2.out_channels, conv_7_params_kws2.out_dim,
        conv_7_params_kws2.stride, 1, conv_7_params_kws2.padding, conv_7_params_kws2.kernel_size,
        conv_7_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_6_out_kws2, (elem_t*)conv_7_w_kws2, (acc_t*)conv_7_b_kws2, (elem_t*)conv_7_out_kws2,

        RELU, conv_7_params_kws2.output_scale, 0,
        conv_7_params_kws2.pool_size, 0, conv_7_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_7_params_kws2.I, conv_7_params_kws2.J,
        conv_7_params_kws2.res_scale,
        conv_7_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_5_out_kws2,
        (elem_t*)conv_7_out_kws2,
        (elem_t*)conv_7_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 

    // conv_8
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_8_params_kws2.batch_size, conv_8_params_kws2.in_dim, conv_8_params_kws2.in_channels,
        conv_8_params_kws2.out_channels, conv_8_params_kws2.out_dim,
        conv_8_params_kws2.stride, 1, conv_8_params_kws2.padding, conv_8_params_kws2.kernel_size,
        conv_8_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_7_out_kws2, (elem_t*)conv_8_w_kws2, (acc_t*)conv_8_b_kws2, (elem_t*)conv_8_out_kws2,

        RELU, conv_8_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_8_params_kws2.pool_size, conv_8_params_kws2.pool_stride, conv_8_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_9_params_kws2.batch_size, conv_9_params_kws2.in_dim, conv_9_params_kws2.in_channels,
        conv_9_params_kws2.out_channels, conv_9_params_kws2.out_dim,
        conv_9_params_kws2.stride, 1, conv_9_params_kws2.padding, conv_9_params_kws2.kernel_size,
        conv_9_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_8_out_kws2, (elem_t*)conv_9_w_kws2, (acc_t*)conv_9_b_kws2, (elem_t*)conv_9_out_kws2,

        RELU, conv_9_params_kws2.output_scale, 0,
        conv_9_params_kws2.pool_size, 0, conv_9_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_9_params_kws2.I, conv_9_params_kws2.J,
        conv_9_params_kws2.res_scale,
        conv_9_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_7_out_kws2,
        (elem_t*)conv_9_out_kws2,
        (elem_t*)conv_9_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
    
    // conv_10
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_10_params_kws2.batch_size, conv_10_params_kws2.in_dim, conv_10_params_kws2.in_channels,
        conv_10_params_kws2.out_channels, conv_10_params_kws2.out_dim,
        conv_10_params_kws2.stride, 1, conv_10_params_kws2.padding, conv_10_params_kws2.kernel_size,
        conv_10_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_9_out_kws2, (elem_t*)conv_10_w_kws2, (acc_t*)conv_10_b_kws2, (elem_t*)conv_10_out_kws2,

        RELU, conv_10_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_10_params_kws2.pool_size, conv_10_params_kws2.pool_stride, conv_10_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_11
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_11_params_kws2.batch_size, conv_11_params_kws2.in_dim, conv_11_params_kws2.in_channels,
        conv_11_params_kws2.out_channels, conv_11_params_kws2.out_dim,
        conv_11_params_kws2.stride, 1, conv_11_params_kws2.padding, conv_11_params_kws2.kernel_size,
        conv_11_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_10_out_kws2, (elem_t*)conv_11_w_kws2, (acc_t*)conv_11_b_kws2, (elem_t*)conv_11_out_kws2,

        RELU, conv_11_params_kws2.output_scale, 0,
        conv_11_params_kws2.pool_size, 0, conv_11_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_11_params_kws2.I, conv_11_params_kws2.J,
        conv_11_params_kws2.res_scale,
        conv_11_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_9_out_kws2,
        (elem_t*)conv_11_out_kws2,
        (elem_t*)conv_11_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 

    // conv_12
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_12_params_kws2.batch_size, conv_12_params_kws2.in_dim, conv_12_params_kws2.in_channels,
        conv_12_params_kws2.out_channels, conv_12_params_kws2.out_dim,
        conv_12_params_kws2.stride, 1, conv_12_params_kws2.padding, conv_12_params_kws2.kernel_size,
        conv_12_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_11_out_kws2, (elem_t*)conv_12_w_kws2, (acc_t*)conv_12_b_kws2, (elem_t*)conv_12_out_kws2,

        RELU, conv_12_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_12_params_kws2.pool_size, conv_12_params_kws2.pool_stride, conv_12_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_13
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_13_params_kws2.batch_size, conv_13_params_kws2.in_dim, conv_13_params_kws2.in_channels,
        conv_13_params_kws2.out_channels, conv_13_params_kws2.out_dim,
        conv_13_params_kws2.stride, 1, conv_13_params_kws2.padding, conv_13_params_kws2.kernel_size,
        conv_13_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_12_out_kws2, (elem_t*)conv_13_w_kws2, (acc_t*)conv_13_b_kws2, (elem_t*)conv_13_out_kws2,

        RELU, conv_13_params_kws2.output_scale, 0,
        conv_13_params_kws2.pool_size, 0, conv_13_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_13_params_kws2.I, conv_13_params_kws2.J,
        conv_13_params_kws2.res_scale,
        conv_13_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_11_out_kws2,
        (elem_t*)conv_13_out_kws2,
        (elem_t*)conv_13_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
    
    // conv_14
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_14_params_kws2.batch_size, conv_14_params_kws2.in_dim, conv_14_params_kws2.in_channels,
        conv_14_params_kws2.out_channels, conv_14_params_kws2.out_dim,
        conv_14_params_kws2.stride, 1, conv_14_params_kws2.padding, conv_14_params_kws2.kernel_size,
        conv_14_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_13_out_kws2, (elem_t*)conv_14_w_kws2, (acc_t*)conv_14_b_kws2, (elem_t*)conv_14_out_kws2,

        RELU, conv_14_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_14_params_kws2.pool_size, conv_14_params_kws2.pool_stride, conv_14_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_15_params_kws2.batch_size, conv_15_params_kws2.in_dim, conv_15_params_kws2.in_channels,
        conv_15_params_kws2.out_channels, conv_15_params_kws2.out_dim,
        conv_15_params_kws2.stride, 1, conv_15_params_kws2.padding, conv_15_params_kws2.kernel_size,
        conv_15_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_14_out_kws2, (elem_t*)conv_15_w_kws2, (acc_t*)conv_15_b_kws2, (elem_t*)conv_15_out_kws2,

        RELU, conv_15_params_kws2.output_scale, 0,
        conv_15_params_kws2.pool_size, 0, conv_15_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_15_params_kws2.I, conv_15_params_kws2.J,
        conv_15_params_kws2.res_scale,
        conv_15_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_13_out_kws2,
        (elem_t*)conv_15_out_kws2,
        (elem_t*)conv_15_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 

    // conv_16
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_16_params_kws2.batch_size, conv_16_params_kws2.in_dim, conv_16_params_kws2.in_channels,
        conv_16_params_kws2.out_channels, conv_16_params_kws2.out_dim,
        conv_16_params_kws2.stride, 1, conv_16_params_kws2.padding, conv_16_params_kws2.kernel_size,
        conv_16_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_15_out_kws2, (elem_t*)conv_16_w_kws2, (acc_t*)conv_16_b_kws2, (elem_t*)conv_16_out_kws2,

        RELU, conv_16_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_16_params_kws2.pool_size, conv_16_params_kws2.pool_stride, conv_16_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_17
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_17_params_kws2.batch_size, conv_17_params_kws2.in_dim, conv_17_params_kws2.in_channels,
        conv_17_params_kws2.out_channels, conv_17_params_kws2.out_dim,
        conv_17_params_kws2.stride, 1, conv_17_params_kws2.padding, conv_17_params_kws2.kernel_size,
        conv_17_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_16_out_kws2, (elem_t*)conv_17_w_kws2, (acc_t*)conv_17_b_kws2, (elem_t*)conv_17_out_kws2,

        RELU, conv_17_params_kws2.output_scale, 0,
        conv_17_params_kws2.pool_size, 0, conv_17_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_17_params_kws2.I, conv_17_params_kws2.J,
        conv_17_params_kws2.res_scale,
        conv_17_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_15_out_kws2,
        (elem_t*)conv_17_out_kws2,
        (elem_t*)conv_17_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
    
    // conv_18
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_18_params_kws2.batch_size, conv_18_params_kws2.in_dim, conv_18_params_kws2.in_channels,
        conv_18_params_kws2.out_channels, conv_18_params_kws2.out_dim,
        conv_18_params_kws2.stride, 1, conv_18_params_kws2.padding, conv_18_params_kws2.kernel_size,
        conv_18_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_17_out_kws2, (elem_t*)conv_18_w_kws2, (acc_t*)conv_18_b_kws2, (elem_t*)conv_18_out_kws2,

        RELU, conv_18_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_18_params_kws2.pool_size, conv_18_params_kws2.pool_stride, conv_18_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_19
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_19_params_kws2.batch_size, conv_19_params_kws2.in_dim, conv_19_params_kws2.in_channels,
        conv_19_params_kws2.out_channels, conv_19_params_kws2.out_dim,
        conv_19_params_kws2.stride, 1, conv_19_params_kws2.padding, conv_19_params_kws2.kernel_size,
        conv_19_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_18_out_kws2, (elem_t*)conv_19_w_kws2, (acc_t*)conv_19_b_kws2, (elem_t*)conv_19_out_kws2,

        RELU, conv_19_params_kws2.output_scale, 0,
        conv_19_params_kws2.pool_size, 0, conv_19_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_19_params_kws2.I, conv_19_params_kws2.J,
        conv_19_params_kws2.res_scale,
        conv_19_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_17_out_kws2,
        (elem_t*)conv_19_out_kws2,
        (elem_t*)conv_19_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 

    // conv_20
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_20_params_kws2.batch_size, conv_20_params_kws2.in_dim, conv_20_params_kws2.in_channels,
        conv_20_params_kws2.out_channels, conv_20_params_kws2.out_dim,
        conv_20_params_kws2.stride, 1, conv_20_params_kws2.padding, conv_20_params_kws2.kernel_size,
        conv_20_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_19_out_kws2, (elem_t*)conv_20_w_kws2, (acc_t*)conv_20_b_kws2, (elem_t*)conv_20_out_kws2,

        RELU, conv_20_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_20_params_kws2.pool_size, conv_20_params_kws2.pool_stride, conv_20_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_21_params_kws2.batch_size, conv_21_params_kws2.in_dim, conv_21_params_kws2.in_channels,
        conv_21_params_kws2.out_channels, conv_21_params_kws2.out_dim,
        conv_21_params_kws2.stride, 1, conv_21_params_kws2.padding, conv_21_params_kws2.kernel_size,
        conv_21_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_20_out_kws2, (elem_t*)conv_21_w_kws2, (acc_t*)conv_21_b_kws2, (elem_t*)conv_21_out_kws2,

        RELU, conv_21_params_kws2.output_scale, 0,
        conv_21_params_kws2.pool_size, 0, conv_21_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_21_params_kws2.I, conv_21_params_kws2.J,
        conv_21_params_kws2.res_scale,
        conv_21_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_19_out_kws2,
        (elem_t*)conv_21_out_kws2,
        (elem_t*)conv_21_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
 

    // conv_22
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_22_params_kws2.batch_size, conv_22_params_kws2.in_dim, conv_22_params_kws2.in_channels,
        conv_22_params_kws2.out_channels, conv_22_params_kws2.out_dim,
        conv_22_params_kws2.stride, 1, conv_22_params_kws2.padding, conv_22_params_kws2.kernel_size,
        conv_22_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_21_out_kws2, (elem_t*)conv_22_w_kws2, (acc_t*)conv_22_b_kws2, (elem_t*)conv_22_out_kws2,

        RELU, conv_22_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_22_params_kws2.pool_size, conv_22_params_kws2.pool_stride, conv_22_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_23
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_23_params_kws2.batch_size, conv_23_params_kws2.in_dim, conv_23_params_kws2.in_channels,
        conv_23_params_kws2.out_channels, conv_23_params_kws2.out_dim,
        conv_23_params_kws2.stride, 1, conv_23_params_kws2.padding, conv_23_params_kws2.kernel_size,
        conv_23_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_22_out_kws2, (elem_t*)conv_23_w_kws2, (acc_t*)conv_23_b_kws2, (elem_t*)conv_23_out_kws2,

        RELU, conv_23_params_kws2.output_scale, 0,
        conv_23_params_kws2.pool_size, 0, conv_23_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // Add residuals
    start = read_cycles();
    tiled_resadd_auto_stride(conv_23_params_kws2.I, conv_23_params_kws2.J,
        conv_23_params_kws2.res_scale,
        conv_23_params_kws2.out_stride,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        (elem_t*)conv_21_out_kws2,
        (elem_t*)conv_23_out_kws2,
        (elem_t*)conv_23_out_kws2,
        true,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_resadd_cycles += end - start;
    resadd_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
    
    // conv_24
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_24_params_kws2.batch_size, conv_24_params_kws2.in_dim, conv_24_params_kws2.in_channels,
        conv_24_params_kws2.out_channels, conv_24_params_kws2.out_dim,
        conv_24_params_kws2.stride, 1, conv_24_params_kws2.padding, conv_24_params_kws2.kernel_size,
        conv_24_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_23_out_kws2, (elem_t*)conv_24_w_kws2, (acc_t*)conv_24_b_kws2, (elem_t*)conv_24_out_kws2,

        RELU, conv_24_params_kws2.output_scale, 0,
        1, 1, 0, false,
    //conv_24_params_kws2.pool_size, conv_24_params_kws2.pool_stride, conv_24_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif
        
    // conv_25
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_25_params_kws2.batch_size, conv_25_params_kws2.in_dim, conv_25_params_kws2.in_channels,
        conv_25_params_kws2.out_channels, conv_25_params_kws2.out_dim,
        conv_25_params_kws2.stride, 1, conv_25_params_kws2.padding, conv_25_params_kws2.kernel_size,
        conv_25_params_kws2.out_stride, 64, 64,

        (elem_t*)conv_24_out_kws2, (elem_t*)conv_25_w_kws2, (acc_t*)conv_25_b_kws2, (elem_t*)conv_25_out_kws2,

        RELU, conv_25_params_kws2.output_scale, 0,
        conv_25_params_kws2.pool_size, 0, conv_25_params_kws2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_kws2);
#endif


    
    for(int i = 0; i < num_cycle; i++){
      if(i < 25){
        cycles[cid][i] = conv_cycles[i];
      }
      else if(i < 26){
        cycles[cid][i] = pool_cycles[i - 25];
      }
      else if (i < 37){
        cycles[cid][i] = resadd_cycles[i - 26];
      }
      else{
        if(i == 37) cycles[cid][i] = total_conv_cycles;
        if(i == 38) cycles[cid][i] = total_pool_cycles;
        if(i == 39) cycles[cid][i] = total_resadd_cycles;
        if(i == 40) cycles[cid][i] = total_conv_cycles + total_pool_cycles + total_resadd_cycles;
      }
    }

    return cycles[cid];
#undef num_cycle
}
