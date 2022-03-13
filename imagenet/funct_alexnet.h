
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "alexnet_orow_params.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* alexnet_function(int cid, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_alex){
#else
uint64_t* alexnet_function(int cid, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (5+3+3+4)

  static uint64_t cycles[num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
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

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
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
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif        
        
    // conv_2
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_2_params.batch_size, conv_2_params.in_dim, conv_2_params.in_channels,
        conv_2_params.out_channels, conv_2_params.out_dim,
        conv_2_params.stride, 1, conv_2_params.padding, conv_2_params.kernel_size,
        conv_2_params.out_stride,

        (elem_t*)conv_1_out_pooled, (elem_t*)conv_2_w, (acc_t*)conv_2_b, (elem_t*)conv_2_out,

        RELU, conv_2_params.output_scale, 0,
        1, 1, 0, false,
	//conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif        
  
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_2_params.batch_size,
        conv_2_params.out_channels, conv_2_params.out_dim, conv_2_params.out_dim_pooled,
        conv_2_params.out_stride,
        conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding,

        (elem_t*)conv_2_out, (elem_t*)conv_2_out_pooled,
	orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif              
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_3_params.batch_size, conv_3_params.in_dim, conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_dim,
        conv_3_params.stride, 1, conv_3_params.padding, conv_3_params.kernel_size,
        conv_3_params.out_stride,

        (elem_t*)conv_2_out_pooled, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

        RELU, conv_3_params.output_scale, 0,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif        
        
    // conv_4
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_4_params.batch_size, conv_4_params.in_dim, conv_4_params.in_channels,
        conv_4_params.out_channels, conv_4_params.out_dim,
        conv_4_params.stride, 1, conv_4_params.padding, conv_4_params.kernel_size,
        conv_4_params.out_stride,

        (elem_t*)conv_3_out, (elem_t*)conv_4_w, (acc_t*)conv_4_b, (elem_t*)conv_4_out,

        RELU, conv_4_params.output_scale, 0,
        conv_4_params.pool_size, 0, conv_4_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif        
        
    // conv_5
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_5_params.batch_size, conv_5_params.in_dim, conv_5_params.in_channels,
        conv_5_params.out_channels, conv_5_params.out_dim,
        conv_5_params.stride, 1, conv_5_params.padding, conv_5_params.kernel_size,
        conv_5_params.out_stride,

        (elem_t*)conv_4_out, (elem_t*)conv_5_w, (acc_t*)conv_5_b, (elem_t*)conv_5_out,

        RELU, conv_5_params.output_scale, 0,
        1, 1, 0, false,
	//conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding, false,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif        
     
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_5_params.batch_size,
        conv_5_params.out_channels, conv_5_params.out_dim, conv_5_params.out_dim_pooled,
        conv_5_params.out_stride,
        conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding,

        (elem_t*)conv_5_out, (elem_t*)conv_5_out_pooled,
	orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif           
    // Global averaging
    
    static elem_t average[1][9216] row_align(MAX_BLOCK_LEN);

    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_5_out_pooled, average, conv_5_params.batch_size,                         
            conv_5_params.out_channels, conv_5_params.out_dim, WS);
       

    end = read_cycles();
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif

    // fc_6
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_6_params.I, fc_6_params.J, fc_6_params.K, fc_6_params.out_stride,
        (elem_t*)average, (elem_t*)fc_6_w, (acc_t*)fc_6_b, (elem_t*)fc_6_out,
        RELU, fc_6_params.output_scale, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif   

    // fc_7
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_7_params.I, fc_7_params.J, fc_7_params.K, fc_7_params.out_stride,
        (elem_t*)fc_6_out, (elem_t*)fc_7_w, (acc_t*)fc_7_b, (elem_t*)fc_7_out,
        RELU, fc_7_params.output_scale, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif   

    // fc_8
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_8_params.I, fc_8_params.J, fc_8_params.K, fc_8_params.out_stride,
        (elem_t*)fc_7_out, (elem_t*)fc_8_w, (acc_t*)fc_8_b, (elem_t*)fc_8_out,
        NO_ACTIVATION, fc_8_params.output_scale, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif   


    for(int i = 0; i < num_layer; i++){
      if(i < 5){
        cycles[i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[i] = total_conv_cycles;
        if(i == 12) cycles[i] = total_matmul_cycles;
        if(i == 13) cycles[i] = total_pool_cycles;
        if(i == 14) cycles[i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles;
#undef num_cycle
}
