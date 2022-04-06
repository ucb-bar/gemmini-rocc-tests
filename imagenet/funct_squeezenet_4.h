
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "squeezenet_params_4.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* squeezenet_function_4(size_t cid, size_t group_id, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_squeeze){
#else
uint64_t* squeezenet_function_4(size_t cid, size_t group_id, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (26+1+3)

  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[26];
    //uint64_t conv_cycles[15];
    uint64_t pool_cycles[1];
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_squeeze4.batch_size, conv_1_params_squeeze4.in_dim, conv_1_params_squeeze4.in_channels,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim,
        conv_1_params_squeeze4.stride, 1, conv_1_params_squeeze4.padding, conv_1_params_squeeze4.kernel_size,
        conv_1_params_squeeze4.out_stride,

        (elem_t*)image4_0, (elem_t*)conv_1_w_squeeze4, (acc_t*)conv_1_b_squeeze4, (elem_t*)conv_1_out_squeeze4,

        RELU, conv_1_params_squeeze4.output_scale, 0,
        1, 0, 0, false,
	//conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding, false,
	WS, orow_divide, batch_divide, cid, group_id, target_util);
        //WS, 2* orow_divide, batch_divide,  cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
//    printf("conv cycles 0: %llu\n", conv_cycles[0]);

#if thread_sync == 1
    pthread_barrier_wait(barrier_squeeze);
#endif        

/*
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_squeeze4.batch_size, conv_1_params_squeeze4.in_dim, conv_1_params_squeeze4.in_channels,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim,
        conv_1_params_squeeze4.stride, 1, conv_1_params_squeeze4.padding, conv_1_params_squeeze4.kernel_size,
        conv_1_params_squeeze4.out_stride,

        (elem_t*)image4_0, (elem_t*)conv_1_w_squeeze4, (acc_t*)conv_1_b_squeeze4, (elem_t*)conv_1_out_squeeze4,

        RELU, conv_1_params_squeeze4.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding, false,

        WS, orow_divide * 2, batch_divide,  orow_divide + cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] += end - start;
#if thread_sync == 1
    pthread_barrier_wait(barrier_squeeze);
#endif   
*/
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_squeeze4.batch_size,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim, conv_1_params_squeeze4.out_dim_pooled,
        conv_1_params_squeeze4.out_stride,
        conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding,

        (elem_t*)conv_1_out_squeeze4, (elem_t*)conv_1_out_squeeze4_pooled,
	orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(barrier_squeeze);
#endif         
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params_squeeze4.I, conv_2_params_squeeze4.J, conv_2_params_squeeze4.K, conv_2_params_squeeze4.out_stride,
        (elem_t*)conv_1_out_squeeze4_pooled, (elem_t*)conv_2_w_squeeze4, (acc_t*)conv_2_b_squeeze4, (elem_t*)conv_2_out_squeeze4,
        RELU, conv_2_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
 
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_3_params_squeeze4.I, conv_3_params_squeeze4.J, conv_3_params_squeeze4.K, conv_3_params_squeeze4.out_stride,
        (elem_t*)conv_2_out_squeeze4, (elem_t*)conv_3_w_squeeze4, (acc_t*)conv_3_b_squeeze4, (elem_t*)conv_4_out_squeeze4,
        RELU, conv_3_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_4
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_4_params_squeeze4.batch_size, conv_4_params_squeeze4.in_dim, conv_4_params_squeeze4.in_channels,
        conv_4_params_squeeze4.out_channels, conv_4_params_squeeze4.out_dim,
        conv_4_params_squeeze4.stride, 1, conv_4_params_squeeze4.padding, conv_4_params_squeeze4.kernel_size, conv_4_params_squeeze4.out_stride,

        (elem_t*)conv_2_out_squeeze4, (elem_t*)conv_4_w_squeeze4, (acc_t*)conv_4_b_squeeze4, (elem_t*)conv_4_out_squeeze4 + conv_4_params_squeeze4.out_channels,

        RELU, conv_4_params_squeeze4.output_scale, 0,
        conv_4_params_squeeze4.pool_size, 0, conv_4_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params_squeeze4.I, conv_5_params_squeeze4.J, conv_5_params_squeeze4.K, conv_5_params_squeeze4.out_stride,
        (elem_t*)conv_4_out_squeeze4, (elem_t*)conv_5_w_squeeze4, (acc_t*)conv_5_b_squeeze4, (elem_t*)conv_5_out_squeeze4,
        RELU, conv_5_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_6
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_6_params_squeeze4.batch_size, conv_6_params_squeeze4.in_dim, conv_6_params_squeeze4.in_channels,
        conv_6_params_squeeze4.out_channels, conv_6_params_squeeze4.out_dim,
        conv_6_params_squeeze4.stride, 1, conv_6_params_squeeze4.padding, conv_6_params_squeeze4.kernel_size, conv_6_params_squeeze4.out_stride,

        (elem_t*)conv_5_out_squeeze4, (elem_t*)conv_6_w_squeeze4, (acc_t*)conv_6_b_squeeze4, (elem_t*)conv_7_out_squeeze4_pooled,

        RELU, conv_6_params_squeeze4.output_scale, 0,
        conv_6_params_squeeze4.pool_size, conv_6_params_squeeze4.pool_stride, conv_6_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_7
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_7_params_squeeze4.batch_size, conv_7_params_squeeze4.in_dim, conv_7_params_squeeze4.in_channels,
        conv_7_params_squeeze4.out_channels, conv_7_params_squeeze4.out_dim,
        conv_7_params_squeeze4.stride, 1, conv_7_params_squeeze4.padding, conv_7_params_squeeze4.kernel_size, conv_7_params_squeeze4.out_stride,

        (elem_t*)conv_5_out_squeeze4, (elem_t*)conv_7_w_squeeze4, (acc_t*)conv_7_b_squeeze4, (elem_t*)conv_7_out_squeeze4_pooled + conv_7_params_squeeze4.out_channels,

        RELU, conv_7_params_squeeze4.output_scale, 0,
        conv_7_params_squeeze4.pool_size, conv_7_params_squeeze4.pool_stride, conv_7_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params_squeeze4.I, conv_8_params_squeeze4.J, conv_8_params_squeeze4.K, conv_8_params_squeeze4.out_stride,
        (elem_t*)conv_7_out_squeeze4_pooled, (elem_t*)conv_8_w_squeeze4, (acc_t*)conv_8_b_squeeze4, (elem_t*)conv_8_out_squeeze4,
        RELU, conv_8_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params_squeeze4.I, conv_9_params_squeeze4.J, conv_9_params_squeeze4.K, conv_9_params_squeeze4.out_stride,
        (elem_t*)conv_8_out_squeeze4, (elem_t*)conv_9_w_squeeze4, (acc_t*)conv_9_b_squeeze4, (elem_t*)conv_10_out_squeeze4,
        RELU, conv_9_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_10
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_10_params_squeeze4.batch_size, conv_10_params_squeeze4.in_dim, conv_10_params_squeeze4.in_channels,
        conv_10_params_squeeze4.out_channels, conv_10_params_squeeze4.out_dim,
        conv_10_params_squeeze4.stride, 1, conv_10_params_squeeze4.padding, conv_10_params_squeeze4.kernel_size, conv_10_params_squeeze4.out_stride,

        (elem_t*)conv_8_out_squeeze4, (elem_t*)conv_10_w_squeeze4, (acc_t*)conv_10_b_squeeze4, (elem_t*)conv_10_out_squeeze4 + conv_10_params_squeeze4.out_channels,

        RELU, conv_10_params_squeeze4.output_scale, 0,
        conv_10_params_squeeze4.pool_size, 0, conv_10_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params_squeeze4.I, conv_11_params_squeeze4.J, conv_11_params_squeeze4.K, conv_11_params_squeeze4.out_stride,
        (elem_t*)conv_10_out_squeeze4, (elem_t*)conv_11_w_squeeze4, (acc_t*)conv_11_b_squeeze4, (elem_t*)conv_11_out_squeeze4,
        RELU, conv_11_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_12
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_12_params_squeeze4.batch_size, conv_12_params_squeeze4.in_dim, conv_12_params_squeeze4.in_channels,
        conv_12_params_squeeze4.out_channels, conv_12_params_squeeze4.out_dim,
        conv_12_params_squeeze4.stride, 1, conv_12_params_squeeze4.padding, conv_12_params_squeeze4.kernel_size, conv_12_params_squeeze4.out_stride,

        (elem_t*)conv_11_out_squeeze4, (elem_t*)conv_12_w_squeeze4, (acc_t*)conv_12_b_squeeze4, (elem_t*)conv_13_out_squeeze4_pooled,

        RELU, conv_12_params_squeeze4.output_scale, 0,
        conv_12_params_squeeze4.pool_size, conv_12_params_squeeze4.pool_stride, conv_12_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_13
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_13_params_squeeze4.batch_size, conv_13_params_squeeze4.in_dim, conv_13_params_squeeze4.in_channels,
        conv_13_params_squeeze4.out_channels, conv_13_params_squeeze4.out_dim,
        conv_13_params_squeeze4.stride, 1, conv_13_params_squeeze4.padding, conv_13_params_squeeze4.kernel_size, conv_13_params_squeeze4.out_stride,

        (elem_t*)conv_11_out_squeeze4, (elem_t*)conv_13_w_squeeze4, (acc_t*)conv_13_b_squeeze4, (elem_t*)conv_13_out_squeeze4_pooled + conv_13_params_squeeze4.out_channels,

        RELU, conv_13_params_squeeze4.output_scale, 0,
        conv_13_params_squeeze4.pool_size, conv_13_params_squeeze4.pool_stride, conv_13_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params_squeeze4.I, conv_14_params_squeeze4.J, conv_14_params_squeeze4.K, conv_14_params_squeeze4.out_stride,
        (elem_t*)conv_13_out_squeeze4_pooled, (elem_t*)conv_14_w_squeeze4, (acc_t*)conv_14_b_squeeze4, (elem_t*)conv_14_out_squeeze4,
        RELU, conv_14_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_15_params_squeeze4.I, conv_15_params_squeeze4.J, conv_15_params_squeeze4.K, conv_15_params_squeeze4.out_stride,
        (elem_t*)conv_14_out_squeeze4, (elem_t*)conv_15_w_squeeze4, (acc_t*)conv_15_b_squeeze4, (elem_t*)conv_16_out_squeeze4,
        RELU, conv_15_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_16
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_16_params_squeeze4.batch_size, 13, 48,
        192, 13,
        1, 1, 1, 3, conv_16_params_squeeze4.out_stride,

        (elem_t*)conv_14_out_squeeze4, (elem_t*)conv_16_w_squeeze4, (acc_t*)conv_16_b_squeeze4, (elem_t*)conv_16_out_squeeze4 + 192,

        RELU, conv_16_params_squeeze4.output_scale, 0,
        conv_16_params_squeeze4.pool_size, 0, conv_16_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);

#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_17
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_17_params_squeeze4.I, conv_17_params_squeeze4.J, conv_17_params_squeeze4.K, conv_17_params_squeeze4.out_stride,
        (elem_t*)conv_16_out_squeeze4, (elem_t*)conv_17_w_squeeze4, (acc_t*)conv_17_b_squeeze4, (elem_t*)conv_17_out_squeeze4,
        RELU, conv_17_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params_squeeze4.I, conv_18_params_squeeze4.J, conv_18_params_squeeze4.K, conv_18_params_squeeze4.out_stride,
        (elem_t*)conv_17_out_squeeze4, (elem_t*)conv_18_w_squeeze4, (acc_t*)conv_18_b_squeeze4, (elem_t*)conv_19_out_squeeze4,
        RELU, conv_18_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_19
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_19_params_squeeze4.batch_size, conv_19_params_squeeze4.in_dim, conv_19_params_squeeze4.in_channels,
        conv_19_params_squeeze4.out_channels, conv_19_params_squeeze4.out_dim,
        conv_19_params_squeeze4.stride, 1, conv_19_params_squeeze4.padding, conv_19_params_squeeze4.kernel_size, conv_19_params_squeeze4.out_stride,

        (elem_t*)conv_17_out_squeeze4, (elem_t*)conv_19_w_squeeze4, (acc_t*)conv_19_b_squeeze4, (elem_t*)conv_19_out_squeeze4 + conv_19_params_squeeze4.out_channels,

        RELU, conv_19_params_squeeze4.output_scale, 0,
        conv_19_params_squeeze4.pool_size, 0, conv_19_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  



    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_20
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_20_params_squeeze4.I, conv_20_params_squeeze4.J, conv_20_params_squeeze4.K, conv_20_params_squeeze4.out_stride,
        (elem_t*)conv_19_out_squeeze4, (elem_t*)conv_20_w_squeeze4, (acc_t*)conv_20_b_squeeze4, (elem_t*)conv_20_out_squeeze4,
        RELU, conv_20_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params_squeeze4.I, conv_21_params_squeeze4.J, conv_21_params_squeeze4.K, conv_21_params_squeeze4.out_stride,
        (elem_t*)conv_20_out_squeeze4, (elem_t*)conv_21_w_squeeze4, (acc_t*)conv_21_b_squeeze4, (elem_t*)conv_22_out_squeeze4,
        RELU, conv_21_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_22
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_22_params_squeeze4.batch_size, conv_22_params_squeeze4.in_dim, conv_22_params_squeeze4.in_channels,
        conv_22_params_squeeze4.out_channels, conv_22_params_squeeze4.out_dim,
        conv_22_params_squeeze4.stride, 1, conv_22_params_squeeze4.padding, conv_22_params_squeeze4.kernel_size, conv_22_params_squeeze4.out_stride,

        (elem_t*)conv_20_out_squeeze4, (elem_t*)conv_22_w_squeeze4, (acc_t*)conv_22_b_squeeze4, (elem_t*)conv_22_out_squeeze4 + conv_22_params_squeeze4.out_channels,

        RELU, conv_22_params_squeeze4.output_scale, 0,
        conv_22_params_squeeze4.pool_size, 0, conv_22_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_23
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_23_params_squeeze4.I, conv_23_params_squeeze4.J, conv_23_params_squeeze4.K, conv_23_params_squeeze4.out_stride,
        (elem_t*)conv_22_out_squeeze4, (elem_t*)conv_23_w_squeeze4, (acc_t*)conv_23_b_squeeze4, (elem_t*)conv_23_out_squeeze4,
        RELU, conv_23_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params_squeeze4.I, conv_24_params_squeeze4.J, conv_24_params_squeeze4.K, conv_24_params_squeeze4.out_stride,
        (elem_t*)conv_23_out_squeeze4, (elem_t*)conv_24_w_squeeze4, (acc_t*)conv_24_b_squeeze4, (elem_t*)conv_25_out_squeeze4,
        RELU, conv_24_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_25
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_25_params_squeeze4.batch_size, conv_25_params_squeeze4.in_dim, conv_25_params_squeeze4.in_channels,
        conv_25_params_squeeze4.out_channels, conv_25_params_squeeze4.out_dim,
        conv_25_params_squeeze4.stride, 1, conv_25_params_squeeze4.padding, conv_25_params_squeeze4.kernel_size, conv_25_params_squeeze4.out_stride,

        (elem_t*)conv_23_out_squeeze4, (elem_t*)conv_25_w_squeeze4, (acc_t*)conv_25_b_squeeze4, (elem_t*)conv_25_out_squeeze4 + conv_25_params_squeeze4.out_channels,

        RELU, conv_25_params_squeeze4.output_scale, 0,
        conv_25_params_squeeze4.pool_size, 0, conv_25_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_26
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_26_params_squeeze4.I, conv_26_params_squeeze4.J, conv_26_params_squeeze4.K, conv_26_params_squeeze4.out_stride,
        (elem_t*)conv_25_out_squeeze4, (elem_t*)conv_26_w_squeeze4, (acc_t*)conv_26_b_squeeze4, (elem_t*)conv_26_out_squeeze4,
        RELU, conv_26_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squeeze);
#endif
  

    for(int i = 0; i < num_cycle; i++){
      if(i < 26){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 27){
        cycles[cid][i] = pool_cycles[i - 26];
      }
      else{
        if(i == 27) cycles[cid][i] = total_conv_cycles;
        if(i == 28) cycles[cid][i] = total_pool_cycles;
        if(i == 29) cycles[cid][i] = total_conv_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}


// single block for squeezenet

uint64_t* squeezenet_block_function_4(size_t cid, size_t group_id, int orow_divide, int batch_divide, int target_util){

#define num_cycle (26+1+3)

  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[26];
    //uint64_t conv_cycles[15];
    uint64_t pool_cycles[1];
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_squeeze4.batch_size, conv_1_params_squeeze4.in_dim, conv_1_params_squeeze4.in_channels,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim,
        conv_1_params_squeeze4.stride, 1, conv_1_params_squeeze4.padding, conv_1_params_squeeze4.kernel_size,
        conv_1_params_squeeze4.out_stride,

        (elem_t*)image4_0, (elem_t*)conv_1_w_squeeze4, (acc_t*)conv_1_b_squeeze4, (elem_t*)conv_1_out_squeeze4,

        RELU, conv_1_params_squeeze4.output_scale, 0,
        1, 0, 0, false,
	//conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding, false,
	WS, orow_divide, batch_divide, cid, group_id, target_util);
        //WS, 2* orow_divide, batch_divide,  cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
//    printf("conv cycles 0: %llu\n", conv_cycles[0]);

#if thread_sync == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif        

/*
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_squeeze4.batch_size, conv_1_params_squeeze4.in_dim, conv_1_params_squeeze4.in_channels,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim,
        conv_1_params_squeeze4.stride, 1, conv_1_params_squeeze4.padding, conv_1_params_squeeze4.kernel_size,
        conv_1_params_squeeze4.out_stride,

        (elem_t*)image4_0, (elem_t*)conv_1_w_squeeze4, (acc_t*)conv_1_b_squeeze4, (elem_t*)conv_1_out_squeeze4,

        RELU, conv_1_params_squeeze4.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding, false,

        WS, orow_divide * 2, batch_divide,  orow_divide + cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] += end - start;
#if thread_sync == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif   
*/
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_squeeze4.batch_size,
        conv_1_params_squeeze4.out_channels, conv_1_params_squeeze4.out_dim, conv_1_params_squeeze4.out_dim_pooled,
        conv_1_params_squeeze4.out_stride,
        conv_1_params_squeeze4.pool_size, conv_1_params_squeeze4.pool_stride, conv_1_params_squeeze4.pool_padding,

        (elem_t*)conv_1_out_squeeze4, (elem_t*)conv_1_out_squeeze4_pooled,
	orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if thread_sync == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif         
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params_squeeze4.I, conv_2_params_squeeze4.J, conv_2_params_squeeze4.K, conv_2_params_squeeze4.out_stride,
        (elem_t*)conv_1_out_squeeze4_pooled, (elem_t*)conv_2_w_squeeze4, (acc_t*)conv_2_b_squeeze4, (elem_t*)conv_2_out_squeeze4,
        RELU, conv_2_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);
//pthread_barrier_wait(barrier_squeeze);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_3_params_squeeze4.I, conv_3_params_squeeze4.J, conv_3_params_squeeze4.K, conv_3_params_squeeze4.out_stride,
        (elem_t*)conv_2_out_squeeze4, (elem_t*)conv_3_w_squeeze4, (acc_t*)conv_3_b_squeeze4, (elem_t*)conv_4_out_squeeze4,
        RELU, conv_3_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_4
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_4_params_squeeze4.batch_size, conv_4_params_squeeze4.in_dim, conv_4_params_squeeze4.in_channels,
        conv_4_params_squeeze4.out_channels, conv_4_params_squeeze4.out_dim,
        conv_4_params_squeeze4.stride, 1, conv_4_params_squeeze4.padding, conv_4_params_squeeze4.kernel_size, conv_4_params_squeeze4.out_stride,

        (elem_t*)conv_2_out_squeeze4, (elem_t*)conv_4_w_squeeze4, (acc_t*)conv_4_b_squeeze4, (elem_t*)conv_4_out_squeeze4 + conv_4_params_squeeze4.out_channels,

        RELU, conv_4_params_squeeze4.output_scale, 0,
        conv_4_params_squeeze4.pool_size, 0, conv_4_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params_squeeze4.I, conv_5_params_squeeze4.J, conv_5_params_squeeze4.K, conv_5_params_squeeze4.out_stride,
        (elem_t*)conv_4_out_squeeze4, (elem_t*)conv_5_w_squeeze4, (acc_t*)conv_5_b_squeeze4, (elem_t*)conv_5_out_squeeze4,
        RELU, conv_5_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_6
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_6_params_squeeze4.batch_size, conv_6_params_squeeze4.in_dim, conv_6_params_squeeze4.in_channels,
        conv_6_params_squeeze4.out_channels, conv_6_params_squeeze4.out_dim,
        conv_6_params_squeeze4.stride, 1, conv_6_params_squeeze4.padding, conv_6_params_squeeze4.kernel_size, conv_6_params_squeeze4.out_stride,

        (elem_t*)conv_5_out_squeeze4, (elem_t*)conv_6_w_squeeze4, (acc_t*)conv_6_b_squeeze4, (elem_t*)conv_7_out_squeeze4_pooled,

        RELU, conv_6_params_squeeze4.output_scale, 0,
        conv_6_params_squeeze4.pool_size, conv_6_params_squeeze4.pool_stride, conv_6_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_7
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_7_params_squeeze4.batch_size, conv_7_params_squeeze4.in_dim, conv_7_params_squeeze4.in_channels,
        conv_7_params_squeeze4.out_channels, conv_7_params_squeeze4.out_dim,
        conv_7_params_squeeze4.stride, 1, conv_7_params_squeeze4.padding, conv_7_params_squeeze4.kernel_size, conv_7_params_squeeze4.out_stride,

        (elem_t*)conv_5_out_squeeze4, (elem_t*)conv_7_w_squeeze4, (acc_t*)conv_7_b_squeeze4, (elem_t*)conv_7_out_squeeze4_pooled + conv_7_params_squeeze4.out_channels,

        RELU, conv_7_params_squeeze4.output_scale, 0,
        conv_7_params_squeeze4.pool_size, conv_7_params_squeeze4.pool_stride, conv_7_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params_squeeze4.I, conv_8_params_squeeze4.J, conv_8_params_squeeze4.K, conv_8_params_squeeze4.out_stride,
        (elem_t*)conv_7_out_squeeze4_pooled, (elem_t*)conv_8_w_squeeze4, (acc_t*)conv_8_b_squeeze4, (elem_t*)conv_8_out_squeeze4,
        RELU, conv_8_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params_squeeze4.I, conv_9_params_squeeze4.J, conv_9_params_squeeze4.K, conv_9_params_squeeze4.out_stride,
        (elem_t*)conv_8_out_squeeze4, (elem_t*)conv_9_w_squeeze4, (acc_t*)conv_9_b_squeeze4, (elem_t*)conv_10_out_squeeze4,
        RELU, conv_9_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_10
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_10_params_squeeze4.batch_size, conv_10_params_squeeze4.in_dim, conv_10_params_squeeze4.in_channels,
        conv_10_params_squeeze4.out_channels, conv_10_params_squeeze4.out_dim,
        conv_10_params_squeeze4.stride, 1, conv_10_params_squeeze4.padding, conv_10_params_squeeze4.kernel_size, conv_10_params_squeeze4.out_stride,

        (elem_t*)conv_8_out_squeeze4, (elem_t*)conv_10_w_squeeze4, (acc_t*)conv_10_b_squeeze4, (elem_t*)conv_10_out_squeeze4 + conv_10_params_squeeze4.out_channels,

        RELU, conv_10_params_squeeze4.output_scale, 0,
        conv_10_params_squeeze4.pool_size, 0, conv_10_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params_squeeze4.I, conv_11_params_squeeze4.J, conv_11_params_squeeze4.K, conv_11_params_squeeze4.out_stride,
        (elem_t*)conv_10_out_squeeze4, (elem_t*)conv_11_w_squeeze4, (acc_t*)conv_11_b_squeeze4, (elem_t*)conv_11_out_squeeze4,
        RELU, conv_11_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_12
     start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_12_params_squeeze4.batch_size, conv_12_params_squeeze4.in_dim, conv_12_params_squeeze4.in_channels,
        conv_12_params_squeeze4.out_channels, conv_12_params_squeeze4.out_dim,
        conv_12_params_squeeze4.stride, 1, conv_12_params_squeeze4.padding, conv_12_params_squeeze4.kernel_size, conv_12_params_squeeze4.out_stride,

        (elem_t*)conv_11_out_squeeze4, (elem_t*)conv_12_w_squeeze4, (acc_t*)conv_12_b_squeeze4, (elem_t*)conv_13_out_squeeze4_pooled,

        RELU, conv_12_params_squeeze4.output_scale, 0,
        conv_12_params_squeeze4.pool_size, conv_12_params_squeeze4.pool_stride, conv_12_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_13
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_13_params_squeeze4.batch_size, conv_13_params_squeeze4.in_dim, conv_13_params_squeeze4.in_channels,
        conv_13_params_squeeze4.out_channels, conv_13_params_squeeze4.out_dim,
        conv_13_params_squeeze4.stride, 1, conv_13_params_squeeze4.padding, conv_13_params_squeeze4.kernel_size, conv_13_params_squeeze4.out_stride,

        (elem_t*)conv_11_out_squeeze4, (elem_t*)conv_13_w_squeeze4, (acc_t*)conv_13_b_squeeze4, (elem_t*)conv_13_out_squeeze4_pooled + conv_13_params_squeeze4.out_channels,

        RELU, conv_13_params_squeeze4.output_scale, 0,
        conv_13_params_squeeze4.pool_size, conv_13_params_squeeze4.pool_stride, conv_13_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params_squeeze4.I, conv_14_params_squeeze4.J, conv_14_params_squeeze4.K, conv_14_params_squeeze4.out_stride,
        (elem_t*)conv_13_out_squeeze4_pooled, (elem_t*)conv_14_w_squeeze4, (acc_t*)conv_14_b_squeeze4, (elem_t*)conv_14_out_squeeze4,
        RELU, conv_14_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_15_params_squeeze4.I, conv_15_params_squeeze4.J, conv_15_params_squeeze4.K, conv_15_params_squeeze4.out_stride,
        (elem_t*)conv_14_out_squeeze4, (elem_t*)conv_15_w_squeeze4, (acc_t*)conv_15_b_squeeze4, (elem_t*)conv_16_out_squeeze4,
        RELU, conv_15_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_16
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_16_params_squeeze4.batch_size, 13, 48,
        192, 13,
        1, 1, 1, 3, conv_16_params_squeeze4.out_stride,

        (elem_t*)conv_14_out_squeeze4, (elem_t*)conv_16_w_squeeze4, (acc_t*)conv_16_b_squeeze4, (elem_t*)conv_16_out_squeeze4 + 192,

        RELU, conv_16_params_squeeze4.output_scale, 0,
        conv_16_params_squeeze4.pool_size, 0, conv_16_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


//pthread_barrier_wait(barrier_squeeze);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_17
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_17_params_squeeze4.I, conv_17_params_squeeze4.J, conv_17_params_squeeze4.K, conv_17_params_squeeze4.out_stride,
        (elem_t*)conv_16_out_squeeze4, (elem_t*)conv_17_w_squeeze4, (acc_t*)conv_17_b_squeeze4, (elem_t*)conv_17_out_squeeze4,
        RELU, conv_17_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params_squeeze4.I, conv_18_params_squeeze4.J, conv_18_params_squeeze4.K, conv_18_params_squeeze4.out_stride,
        (elem_t*)conv_17_out_squeeze4, (elem_t*)conv_18_w_squeeze4, (acc_t*)conv_18_b_squeeze4, (elem_t*)conv_19_out_squeeze4,
        RELU, conv_18_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_19
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_19_params_squeeze4.batch_size, conv_19_params_squeeze4.in_dim, conv_19_params_squeeze4.in_channels,
        conv_19_params_squeeze4.out_channels, conv_19_params_squeeze4.out_dim,
        conv_19_params_squeeze4.stride, 1, conv_19_params_squeeze4.padding, conv_19_params_squeeze4.kernel_size, conv_19_params_squeeze4.out_stride,

        (elem_t*)conv_17_out_squeeze4, (elem_t*)conv_19_w_squeeze4, (acc_t*)conv_19_b_squeeze4, (elem_t*)conv_19_out_squeeze4 + conv_19_params_squeeze4.out_channels,

        RELU, conv_19_params_squeeze4.output_scale, 0,
        conv_19_params_squeeze4.pool_size, 0, conv_19_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


//pthread_barrier_wait(barrier_squeeze);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_20
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_20_params_squeeze4.I, conv_20_params_squeeze4.J, conv_20_params_squeeze4.K, conv_20_params_squeeze4.out_stride,
        (elem_t*)conv_19_out_squeeze4, (elem_t*)conv_20_w_squeeze4, (acc_t*)conv_20_b_squeeze4, (elem_t*)conv_20_out_squeeze4,
        RELU, conv_20_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params_squeeze4.I, conv_21_params_squeeze4.J, conv_21_params_squeeze4.K, conv_21_params_squeeze4.out_stride,
        (elem_t*)conv_20_out_squeeze4, (elem_t*)conv_21_w_squeeze4, (acc_t*)conv_21_b_squeeze4, (elem_t*)conv_22_out_squeeze4,
        RELU, conv_21_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_22
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_22_params_squeeze4.batch_size, conv_22_params_squeeze4.in_dim, conv_22_params_squeeze4.in_channels,
        conv_22_params_squeeze4.out_channels, conv_22_params_squeeze4.out_dim,
        conv_22_params_squeeze4.stride, 1, conv_22_params_squeeze4.padding, conv_22_params_squeeze4.kernel_size, conv_22_params_squeeze4.out_stride,

        (elem_t*)conv_20_out_squeeze4, (elem_t*)conv_22_w_squeeze4, (acc_t*)conv_22_b_squeeze4, (elem_t*)conv_22_out_squeeze4 + conv_22_params_squeeze4.out_channels,

        RELU, conv_22_params_squeeze4.output_scale, 0,
        conv_22_params_squeeze4.pool_size, 0, conv_22_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_23
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_23_params_squeeze4.I, conv_23_params_squeeze4.J, conv_23_params_squeeze4.K, conv_23_params_squeeze4.out_stride,
        (elem_t*)conv_22_out_squeeze4, (elem_t*)conv_23_w_squeeze4, (acc_t*)conv_23_b_squeeze4, (elem_t*)conv_23_out_squeeze4,
        RELU, conv_23_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params_squeeze4.I, conv_24_params_squeeze4.J, conv_24_params_squeeze4.K, conv_24_params_squeeze4.out_stride,
        (elem_t*)conv_23_out_squeeze4, (elem_t*)conv_24_w_squeeze4, (acc_t*)conv_24_b_squeeze4, (elem_t*)conv_25_out_squeeze4,
        RELU, conv_24_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
        


    // conv_25
     start = read_cycles();

    tiled_conv_A_stride_auto_cid(
        conv_25_params_squeeze4.batch_size, conv_25_params_squeeze4.in_dim, conv_25_params_squeeze4.in_channels,
        conv_25_params_squeeze4.out_channels, conv_25_params_squeeze4.out_dim,
        conv_25_params_squeeze4.stride, 1, conv_25_params_squeeze4.padding, conv_25_params_squeeze4.kernel_size, conv_25_params_squeeze4.out_stride,

        (elem_t*)conv_23_out_squeeze4, (elem_t*)conv_25_w_squeeze4, (acc_t*)conv_25_b_squeeze4, (elem_t*)conv_25_out_squeeze4 + conv_25_params_squeeze4.out_channels,

        RELU, conv_25_params_squeeze4.output_scale, 0,
        conv_25_params_squeeze4.pool_size, 0, conv_25_params_squeeze4.pool_padding, false,

        WS, orow_divide, batch_divide, cid, group_id, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif  


    // conv_26
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_26_params_squeeze4.I, conv_26_params_squeeze4.J, conv_26_params_squeeze4.K, conv_26_params_squeeze4.out_stride,
        (elem_t*)conv_25_out_squeeze4, (elem_t*)conv_26_w_squeeze4, (acc_t*)conv_26_b_squeeze4, (elem_t*)conv_26_out_squeeze4,
        RELU, conv_26_params_squeeze4.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, group_id, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;

#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_squeeze);
#endif
  

    for(int i = 0; i < num_cycle; i++){
      if(i < 26){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 27){
        cycles[cid][i] = pool_cycles[i - 26];
      }
      else{
        if(i == 27) cycles[cid][i] = total_conv_cycles;
        if(i == 28) cycles[cid][i] = total_pool_cycles;
        if(i == 29) cycles[cid][i] = total_conv_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}
