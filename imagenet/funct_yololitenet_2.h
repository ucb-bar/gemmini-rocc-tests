
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "yololitenet_params_2.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* yololitenet_function_2(int cid, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_yololite){
#else
uint64_t* yololitenet_function_2(int cid, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (7+5+3)

  static uint64_t cycles[num_proc][num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[7];
    uint64_t pool_cycles[5];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif

        
    // conv_4
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_1_params_yololite2.batch_size, conv_1_params_yololite2.in_dim, conv_1_params_yololite2.in_channels,
        conv_1_params_yololite2.out_channels, conv_1_params_yololite2.out_dim,
        conv_1_params_yololite2.stride, 1, conv_1_params_yololite2.padding, conv_1_params_yololite2.kernel_size,
        conv_1_params_yololite2.out_stride, conv_1_params_yololite2.in_channels, 64,

        (elem_t*)images_2, (elem_t*)conv_1_w_yololite2, (acc_t*)conv_1_b_yololite2, (elem_t*)conv_1_out_yololite2,

        RELU, conv_1_params_yololite2.output_scale, 0,
        1, 1, 0, false,
	//conv_1_params_yololite2.pool_size, conv_1_params_yololite2.pool_stride, conv_1_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif        
 
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params_yololite2.batch_size,
        conv_1_params_yololite2.out_channels, conv_1_params_yololite2.out_dim, conv_1_params_yololite2.out_dim_pooled,
        conv_1_params_yololite2.out_stride,
        conv_1_params_yololite2.pool_size, conv_1_params_yololite2.pool_stride, conv_1_params_yololite2.pool_padding,

        (elem_t*)conv_1_out_yololite2, (elem_t*)conv_1_out_yololite2_pooled,
	orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif        
        
    // conv_2
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_2_params_yololite2.batch_size, conv_2_params_yololite2.in_dim, conv_2_params_yololite2.in_channels,
        conv_2_params_yololite2.out_channels, conv_2_params_yololite2.out_dim,
        conv_2_params_yololite2.stride, 1, conv_2_params_yololite2.padding, conv_2_params_yololite2.kernel_size,
        conv_2_params_yololite2.out_stride, 64, 64,

        (elem_t*)conv_1_out_yololite2_pooled, (elem_t*)conv_2_w_yololite2, (acc_t*)conv_2_b_yololite2, (elem_t*)conv_2_out_yololite2,

        RELU, conv_2_params_yololite2.output_scale, 0,
        1, 1, 0, false,
	//conv_2_params_yololite2.pool_size, conv_2_params_yololite2.pool_stride, conv_2_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif        
  
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_2_params_yololite2.batch_size,
        conv_2_params_yololite2.out_channels, conv_2_params_yololite2.out_dim, conv_2_params_yololite2.out_dim_pooled,
        conv_2_params_yololite2.out_stride,
        conv_2_params_yololite2.pool_size, conv_2_params_yololite2.pool_stride, conv_2_params_yololite2.pool_padding,

        (elem_t*)conv_2_out_yololite2, (elem_t*)conv_2_out_yololite2_pooled,
	orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif              

    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_stride(
        conv_3_params_yololite2.batch_size, conv_3_params_yololite2.in_dim, conv_3_params_yololite2.in_channels,
        conv_3_params_yololite2.out_channels, conv_3_params_yololite2.out_dim,
        conv_3_params_yololite2.stride, 1, conv_3_params_yololite2.padding, conv_3_params_yololite2.kernel_size,
        conv_3_params_yololite2.out_stride, 64, 64,

        (elem_t*)conv_2_out_yololite2_pooled, (elem_t*)conv_3_w_yololite2, (acc_t*)conv_3_b_yololite2, (elem_t*)conv_3_out_yololite2,

        RELU, conv_3_params_yololite2.output_scale, 0,
        1, 1, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
 
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_3_params_yololite2.batch_size,
        conv_3_params_yololite2.out_channels, conv_3_params_yololite2.out_dim, conv_3_params_yololite2.out_dim_pooled,
        conv_3_params_yololite2.out_stride,
        conv_3_params_yololite2.pool_size, conv_3_params_yololite2.pool_stride, conv_3_params_yololite2.pool_padding,

        (elem_t*)conv_3_out_yololite2, (elem_t*)conv_3_out_yololite2_pooled,
    orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
        
    // conv_4
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_4_params_yololite2.batch_size, conv_4_params_yololite2.in_dim, conv_4_params_yololite2.in_channels,
        conv_4_params_yololite2.out_channels, conv_4_params_yololite2.out_dim,
        conv_4_params_yololite2.stride, 1, conv_4_params_yololite2.padding, conv_4_params_yololite2.kernel_size,
        conv_4_params_yololite2.out_stride,

        (elem_t*)conv_3_out_yololite2_pooled, (elem_t*)conv_4_w_yololite2, (acc_t*)conv_4_b_yololite2, (elem_t*)conv_4_out_yololite2,

        RELU, conv_4_params_yololite2.output_scale, 0,
        1, 1, 0, false,
    //conv_2_params_yololite2.pool_size, conv_2_params_yololite2.pool_stride, conv_2_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
  
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_4_params_yololite2.batch_size,
        conv_4_params_yololite2.out_channels, conv_4_params_yololite2.out_dim, conv_4_params_yololite2.out_dim_pooled,
        conv_4_params_yololite2.out_stride,
        conv_4_params_yololite2.pool_size, conv_4_params_yololite2.pool_stride, conv_4_params_yololite2.pool_padding,

        (elem_t*)conv_4_out_yololite2, (elem_t*)conv_4_out_yololite2_pooled,
    orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
    
    // conv_5
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_5_params_yololite2.batch_size, conv_5_params_yololite2.in_dim, conv_5_params_yololite2.in_channels,
        conv_5_params_yololite2.out_channels, conv_5_params_yololite2.out_dim,
        conv_5_params_yololite2.stride, 1, conv_5_params_yololite2.padding, conv_5_params_yololite2.kernel_size,
        conv_5_params_yololite2.out_stride,

        (elem_t*)conv_4_out_yololite2_pooled, (elem_t*)conv_5_w_yololite2, (acc_t*)conv_5_b_yololite2, (elem_t*)conv_5_out_yololite2,

        RELU, conv_5_params_yololite2.output_scale, 0,
        1, 1, 0, false,
    //conv_1_params_yololite2.pool_size, conv_1_params_yololite2.pool_stride, conv_1_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
 
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_5_params_yololite2.batch_size,
        conv_5_params_yololite2.out_channels, conv_5_params_yololite2.out_dim, conv_5_params_yololite2.out_dim_pooled,
        conv_5_params_yololite2.out_stride,
        conv_5_params_yololite2.pool_size, conv_5_params_yololite2.pool_stride, conv_5_params_yololite2.pool_padding,

        (elem_t*)conv_5_out_yololite2, (elem_t*)conv_5_out_yololite2_pooled,
    orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
        
    // conv_6
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_6_params_yololite2.batch_size, conv_6_params_yololite2.in_dim, conv_6_params_yololite2.in_channels,
        conv_6_params_yololite2.out_channels, conv_6_params_yololite2.out_dim,
        conv_6_params_yololite2.stride, 1, conv_6_params_yololite2.padding, conv_6_params_yololite2.kernel_size,
        conv_6_params_yololite2.out_stride,

        (elem_t*)conv_5_out_yololite2_pooled, (elem_t*)conv_6_w_yololite2, (acc_t*)conv_6_b_yololite2, (elem_t*)conv_6_out_yololite2,

        RELU, conv_6_params_yololite2.output_scale, 0,
        1, 1, 0, false,
    //conv_2_params_yololite2.pool_size, conv_2_params_yololite2.pool_stride, conv_2_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
  
    // conv_7
    start = read_cycles();
    /*
    tiled_conv_A_stride_auto_cid(
        conv_7_params_yololite2.batch_size, conv_7_params_yololite2.in_dim, conv_7_params_yololite2.in_channels,
        conv_7_params_yololite2.out_channels, conv_7_params_yololite2.out_dim,
        conv_7_params_yololite2.stride, 1, conv_7_params_yololite2.padding, conv_7_params_yololite2.kernel_size,
        conv_7_params_yololite2.out_stride,

        (elem_t*)conv_6_out_yololite2, (elem_t*)conv_7_w_yololite2, (acc_t*)conv_7_b_yololite2, (elem_t*)conv_7_out_yololite2,

        RELU, conv_7_params_yololite2.output_scale, 0,
        1, 1, 0, false,
    //conv_7_params_yololite2.pool_size, conv_7_params_yololite2.pool_stride, conv_7_params_yololite2.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);
*/
    tiled_matmul_nn_auto_cid(conv_7_params_yololite2.I, conv_7_params_yololite2.J, conv_7_params_yololite2.K, conv_7_params_yololite2.out_stride,
            (elem_t*)conv_6_out_yololite2, (elem_t*)conv_7_w_yololite2, (acc_t*)conv_7_b_yololite2, (elem_t*)conv_7_out_yololite2,
            RELU, conv_7_params_yololite2.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, target_util);

    
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_yololite);
#endif
 

    for(int i = 0; i < num_cycle; i++){
      if(i < 7){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 12){
        cycles[cid][i] = pool_cycles[i - 7];
      }
      else{
        if(i == 12) cycles[cid][i] = total_conv_cycles;
        if(i == 13) cycles[cid][i] = total_pool_cycles;
        if(i == 14) cycles[cid][i] = total_conv_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}
