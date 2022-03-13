
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "squeezenet_orow_params.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* squeezenet_function(int cid, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_squeeze){
#else
uint64_t* squeezenet_function(int cid, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (12+15+1+4)

  static uint64_t cycles[num_cycle];
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[12];
    uint64_t matmul_cycles[15];
    uint64_t pool_cycles[1];

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

        WS, 2* orow_divide, batch_divide,  cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;

#if thread_sync == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide * 2, batch_divide,  orow_divide + cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(barrier_squ);
#endif   
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params.batch_size,
        conv_1_params.out_channels, conv_1_params.out_dim, conv_1_params.out_dim_pooled,
        conv_1_params.out_stride,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

        (elem_t*)conv_1_out, (elem_t*)conv_1_out_pooled,
	orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if thread_sync == 1
    pthread_barrier_wait(barrier_squ);
#endif         
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params.I, conv_2_params.J, conv_2_params.K, conv_2_params.out_stride,
        (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,
        RELU, conv_2_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);
//pthread_barrier_wait(barrier_squ);


    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_3_params.I, conv_3_params.J, conv_3_params.K, conv_3_params.out_stride,
        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_4_out,
        RELU, conv_3_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params.I, conv_5_params.J, conv_5_params.K, conv_5_params.out_stride,
        (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,
        RELU, conv_5_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_8
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_8_params.I, conv_8_params.J, conv_8_params.K, conv_8_params.out_stride,
        (elem_t*)conv_7_out_pooled, (elem_t*)conv_8_w, (acc_t*)conv_8_b, (elem_t*)conv_8_out,
        RELU, conv_8_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_9
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_9_params.I, conv_9_params.J, conv_9_params.K, conv_9_params.out_stride,
        (elem_t*)conv_8_out, (elem_t*)conv_9_w, (acc_t*)conv_9_b, (elem_t*)conv_10_out,
        RELU, conv_9_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_11
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_11_params.I, conv_11_params.J, conv_11_params.K, conv_11_params.out_stride,
        (elem_t*)conv_10_out, (elem_t*)conv_11_w, (acc_t*)conv_11_b, (elem_t*)conv_11_out,
        RELU, conv_11_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_14
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_14_params.I, conv_14_params.J, conv_14_params.K, conv_14_params.out_stride,
        (elem_t*)conv_13_out_pooled, (elem_t*)conv_14_w, (acc_t*)conv_14_b, (elem_t*)conv_14_out,
        RELU, conv_14_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_15
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_15_params.I, conv_15_params.J, conv_15_params.K, conv_15_params.out_stride,
        (elem_t*)conv_14_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_16_out,
        RELU, conv_15_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


//pthread_barrier_wait(barrier_squ);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_17
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_17_params.I, conv_17_params.J, conv_17_params.K, conv_17_params.out_stride,
        (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,
        RELU, conv_17_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_18
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_18_params.I, conv_18_params.J, conv_18_params.K, conv_18_params.out_stride,
        (elem_t*)conv_17_out, (elem_t*)conv_18_w, (acc_t*)conv_18_b, (elem_t*)conv_19_out,
        RELU, conv_18_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


//pthread_barrier_wait(barrier_squ);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_20
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_20_params.I, conv_20_params.J, conv_20_params.K, conv_20_params.out_stride,
        (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,
        RELU, conv_20_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_21
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_21_params.I, conv_21_params.J, conv_21_params.K, conv_21_params.out_stride,
        (elem_t*)conv_20_out, (elem_t*)conv_21_w, (acc_t*)conv_21_b, (elem_t*)conv_22_out,
        RELU, conv_21_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_23
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_23_params.I, conv_23_params.J, conv_23_params.K, conv_23_params.out_stride,
        (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,
        RELU, conv_23_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
        
    // conv_24
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_24_params.I, conv_24_params.J, conv_24_params.K, conv_24_params.out_stride,
        (elem_t*)conv_23_out, (elem_t*)conv_24_w, (acc_t*)conv_24_b, (elem_t*)conv_25_out,
        RELU, conv_24_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
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

        WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif  


    // conv_26
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_26_params.I, conv_26_params.J, conv_26_params.K, conv_26_params.out_stride,
        (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,
        RELU, conv_26_params.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_squ);
#endif
  

    for(int i = 0; i < num_count; i++){
      if(i < 12){
        cycles[i] = conv_cycles[i];
      }
      else if(i < 27){
        cycles[i] = matmul_cycles[i - 5];
      }
      else if (i < 28){
        cycles[i] = pool_cycles[i - 8];
      }
      else{
        if(i == 28) cycles[i] = total_conv_cycles;
        if(i == 29) cycles[i] = total_matmul_cycles;
        if(i == 30) cycles[i] = total_pool_cycles;
        if(i == 31) cycles[i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}
