
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "alexnet_params_8.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* alexnet_function_8(size_t cid, size_t group_id, bool part1, bool part2, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_alex){
#else
uint64_t* alexnet_function_8(size_t cid, size_t group_id, bool part1, bool part2, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (5+3+3+4)

  static uint64_t cycles[NUM_CORE][num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif

    if(part1){
      // conv_1
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_1_params_alex8.batch_size, conv_1_params_alex8.in_dim, conv_1_params_alex8.in_channels,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim,
          conv_1_params_alex8.stride, 1, conv_1_params_alex8.padding, conv_1_params_alex8.kernel_size,
          conv_1_params_alex8.out_stride,

          (elem_t*)image8_4, (elem_t*)conv_1_w_alex8, (acc_t*)conv_1_b_alex8, (elem_t*)conv_1_out_alex8,

          RELU, conv_1_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
   
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_1_params_alex8.batch_size,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim, conv_1_params_alex8.out_dim_pooled,
          conv_1_params_alex8.out_stride,
          conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding,

          (elem_t*)conv_1_out_alex8, (elem_t*)conv_1_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_2
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_2_params_alex8.batch_size, conv_2_params_alex8.in_dim, conv_2_params_alex8.in_channels,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim,
          conv_2_params_alex8.stride, 1, conv_2_params_alex8.padding, conv_2_params_alex8.kernel_size,
          conv_2_params_alex8.out_stride,

          (elem_t*)conv_1_out_alex8_pooled, (elem_t*)conv_2_w_alex8, (acc_t*)conv_2_b_alex8, (elem_t*)conv_2_out_alex8,

          RELU, conv_2_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
    
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_2_params_alex8.batch_size,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim, conv_2_params_alex8.out_dim_pooled,
          conv_2_params_alex8.out_stride,
          conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding,

          (elem_t*)conv_2_out_alex8, (elem_t*)conv_2_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += (end > start ? end - start : 0);
      pool_cycles[1] = (end > start ? end - start : 0);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif              
      // conv_3
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_3_params_alex8.batch_size, conv_3_params_alex8.in_dim, conv_3_params_alex8.in_channels,
          conv_3_params_alex8.out_channels, conv_3_params_alex8.out_dim,
          conv_3_params_alex8.stride, 1, conv_3_params_alex8.padding, conv_3_params_alex8.kernel_size,
          conv_3_params_alex8.out_stride,

          (elem_t*)conv_2_out_alex8_pooled, (elem_t*)conv_3_w_alex8, (acc_t*)conv_3_b_alex8, (elem_t*)conv_3_out_alex8,

          RELU, conv_3_params_alex8.output_scale, 0,
          conv_3_params_alex8.pool_size, 0, conv_3_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_4_params_alex8.batch_size, conv_4_params_alex8.in_dim, conv_4_params_alex8.in_channels,
          conv_4_params_alex8.out_channels, conv_4_params_alex8.out_dim,
          conv_4_params_alex8.stride, 1, conv_4_params_alex8.padding, conv_4_params_alex8.kernel_size,
          conv_4_params_alex8.out_stride,

          (elem_t*)conv_3_out_alex8, (elem_t*)conv_4_w_alex8, (acc_t*)conv_4_b_alex8, (elem_t*)conv_4_out_alex8,

          RELU, conv_4_params_alex8.output_scale, 0,
          conv_4_params_alex8.pool_size, 0, conv_4_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_5
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_5_params_alex8.batch_size, conv_5_params_alex8.in_dim, conv_5_params_alex8.in_channels,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim,
          conv_5_params_alex8.stride, 1, conv_5_params_alex8.padding, conv_5_params_alex8.kernel_size,
          conv_5_params_alex8.out_stride,

          (elem_t*)conv_4_out_alex8, (elem_t*)conv_5_w_alex8, (acc_t*)conv_5_b_alex8, (elem_t*)conv_5_out_alex8,

          RELU, conv_5_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(part2){
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_5_params_alex8.batch_size,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, conv_5_params_alex8.out_dim_pooled,
          conv_5_params_alex8.out_stride,
          conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding,

          (elem_t*)conv_5_out_alex8, (elem_t*)conv_5_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif           
      // Global averaging
      
      static elem_t average[8][9216] row_align(MAX_BLOCK_LEN);

      start = read_cycles();
      if(cid == 0)
          tiled_global_average_auto(conv_5_out_alex8_pooled, average, conv_5_params_alex8.batch_size,                         
              conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif

      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8, (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8,
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8 + (int)(fc_6_params_alex8.J / 2), (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8 + (int)(fc_6_params_alex8.J / 2),
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] += end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   


      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8, (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8,
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8 + (int)(fc_7_params_alex8.J / 2), (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8 + (int)(fc_7_params_alex8.J / 2),
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] += end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      
      // fc_8
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_8_params_alex8.I, fc_8_params_alex8.J, fc_8_params_alex8.K, fc_8_params_alex8.out_stride,
          (elem_t*)fc_7_out_alex8, (elem_t*)fc_8_w_alex8, (acc_t*)fc_8_b_alex8, (elem_t*)fc_8_out_alex8,
          NO_ACTIVATION, fc_8_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 5){
        cycles[cid][i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[cid][i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[cid][i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[cid][i] = total_conv_cycles;
        if(i == 12) cycles[cid][i] = total_matmul_cycles;
        if(i == 13) cycles[cid][i] = total_pool_cycles;
        if(i == 14) cycles[cid][i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}

uint64_t* alexnet_block_function_8(size_t cid, size_t group_id, bool part1, bool part2, int orow_divide, int batch_divide, int target_util){
#define num_cycle (5+3+3+4)

  static uint64_t cycles[NUM_CORE][num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_alex);
#endif

    if(part1){
      // conv_1
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_1_params_alex8.batch_size, conv_1_params_alex8.in_dim, conv_1_params_alex8.in_channels,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim,
          conv_1_params_alex8.stride, 1, conv_1_params_alex8.padding, conv_1_params_alex8.kernel_size,
          conv_1_params_alex8.out_stride,

          (elem_t*)image8_4, (elem_t*)conv_1_w_alex8, (acc_t*)conv_1_b_alex8, (elem_t*)conv_1_out_alex8,

          RELU, conv_1_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif        
   
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_1_params_alex8.batch_size,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim, conv_1_params_alex8.out_dim_pooled,
          conv_1_params_alex8.out_stride,
          conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding,

          (elem_t*)conv_1_out_alex8, (elem_t*)conv_1_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_2
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_2_params_alex8.batch_size, conv_2_params_alex8.in_dim, conv_2_params_alex8.in_channels,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim,
          conv_2_params_alex8.stride, 1, conv_2_params_alex8.padding, conv_2_params_alex8.kernel_size,
          conv_2_params_alex8.out_stride,

          (elem_t*)conv_1_out_alex8_pooled, (elem_t*)conv_2_w_alex8, (acc_t*)conv_2_b_alex8, (elem_t*)conv_2_out_alex8,

          RELU, conv_2_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif        
    
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_2_params_alex8.batch_size,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim, conv_2_params_alex8.out_dim_pooled,
          conv_2_params_alex8.out_stride,
          conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding,

          (elem_t*)conv_2_out_alex8, (elem_t*)conv_2_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += (end > start ? end - start : 0);
      pool_cycles[1] = (end > start ? end - start : 0);
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif              
      // conv_3
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_3_params_alex8.batch_size, conv_3_params_alex8.in_dim, conv_3_params_alex8.in_channels,
          conv_3_params_alex8.out_channels, conv_3_params_alex8.out_dim,
          conv_3_params_alex8.stride, 1, conv_3_params_alex8.padding, conv_3_params_alex8.kernel_size,
          conv_3_params_alex8.out_stride,

          (elem_t*)conv_2_out_alex8_pooled, (elem_t*)conv_3_w_alex8, (acc_t*)conv_3_b_alex8, (elem_t*)conv_3_out_alex8,

          RELU, conv_3_params_alex8.output_scale, 0,
          conv_3_params_alex8.pool_size, 0, conv_3_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_4_params_alex8.batch_size, conv_4_params_alex8.in_dim, conv_4_params_alex8.in_channels,
          conv_4_params_alex8.out_channels, conv_4_params_alex8.out_dim,
          conv_4_params_alex8.stride, 1, conv_4_params_alex8.padding, conv_4_params_alex8.kernel_size,
          conv_4_params_alex8.out_stride,

          (elem_t*)conv_3_out_alex8, (elem_t*)conv_4_w_alex8, (acc_t*)conv_4_b_alex8, (elem_t*)conv_4_out_alex8,

          RELU, conv_4_params_alex8.output_scale, 0,
          conv_4_params_alex8.pool_size, 0, conv_4_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_5
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_5_params_alex8.batch_size, conv_5_params_alex8.in_dim, conv_5_params_alex8.in_channels,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim,
          conv_5_params_alex8.stride, 1, conv_5_params_alex8.padding, conv_5_params_alex8.kernel_size,
          conv_5_params_alex8.out_stride,

          (elem_t*)conv_4_out_alex8, (elem_t*)conv_5_w_alex8, (acc_t*)conv_5_b_alex8, (elem_t*)conv_5_out_alex8,

          RELU, conv_5_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif
    }

    if(part2){
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_5_params_alex8.batch_size,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, conv_5_params_alex8.out_dim_pooled,
          conv_5_params_alex8.out_stride,
          conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding,

          (elem_t*)conv_5_out_alex8, (elem_t*)conv_5_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  //gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif           
      // Global averaging
      
      static elem_t average[8][9216] row_align(MAX_BLOCK_LEN);

      start = read_cycles();
      if(cid == 0)
          tiled_global_average_auto(conv_5_out_alex8_pooled, average, conv_5_params_alex8.batch_size,                         
              conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif

      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8, (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8,
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif   
      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8 + (int)(fc_6_params_alex8.J / 2), (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8 + (int)(fc_6_params_alex8.J / 2),
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] += end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif   


      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8, (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8,
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif   
      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8 + (int)(fc_7_params_alex8.J / 2), (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8 + (int)(fc_7_params_alex8.J / 2),
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] += end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif   
       

      // fc_8
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_8_params_alex8.I, fc_8_params_alex8.J, fc_8_params_alex8.K, fc_8_params_alex8.out_stride,
          (elem_t*)fc_7_out_alex8, (elem_t*)fc_8_w_alex8, (acc_t*)fc_8_b_alex8, (elem_t*)fc_8_out_alex8,
          NO_ACTIVATION, fc_8_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_alex);
#endif   
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 5){
        cycles[cid][i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[cid][i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[cid][i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[cid][i] = total_conv_cycles;
        if(i == 12) cycles[cid][i] = total_matmul_cycles;
        if(i == 13) cycles[cid][i] = total_pool_cycles;
        if(i == 14) cycles[cid][i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}

uint64_t* alexnet_batch_function_8(size_t cid, size_t group_id, bool part1, bool part2, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_alex){
#define num_cycle (5+3+3+4)

  static uint64_t cycles[NUM_CORE][num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[5];
    uint64_t matmul_cycles[3];
    uint64_t pool_cycles[3];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_alex);
#endif

    int target_util_save = target_util;
    if(part1){
	target_util = -1;
      // conv_1
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_1_params_alex8.batch_size, conv_1_params_alex8.in_dim, conv_1_params_alex8.in_channels,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim,
          conv_1_params_alex8.stride, 1, conv_1_params_alex8.padding, conv_1_params_alex8.kernel_size,
          conv_1_params_alex8.out_stride,

          (elem_t*)image8_4, (elem_t*)conv_1_w_alex8, (acc_t*)conv_1_b_alex8, (elem_t*)conv_1_out_alex8,

          RELU, conv_1_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
   
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_1_params_alex8.batch_size,
          conv_1_params_alex8.out_channels, conv_1_params_alex8.out_dim, conv_1_params_alex8.out_dim_pooled,
          conv_1_params_alex8.out_stride,
          conv_1_params_alex8.pool_size, conv_1_params_alex8.pool_stride, conv_1_params_alex8.pool_padding,

          (elem_t*)conv_1_out_alex8, (elem_t*)conv_1_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_2
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_2_params_alex8.batch_size, conv_2_params_alex8.in_dim, conv_2_params_alex8.in_channels,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim,
          conv_2_params_alex8.stride, 1, conv_2_params_alex8.padding, conv_2_params_alex8.kernel_size,
          conv_2_params_alex8.out_stride,

          (elem_t*)conv_1_out_alex8_pooled, (elem_t*)conv_2_w_alex8, (acc_t*)conv_2_b_alex8, (elem_t*)conv_2_out_alex8,

          RELU, conv_2_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
    
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_2_params_alex8.batch_size,
          conv_2_params_alex8.out_channels, conv_2_params_alex8.out_dim, conv_2_params_alex8.out_dim_pooled,
          conv_2_params_alex8.out_stride,
          conv_2_params_alex8.pool_size, conv_2_params_alex8.pool_stride, conv_2_params_alex8.pool_padding,

          (elem_t*)conv_2_out_alex8, (elem_t*)conv_2_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += (end > start ? end - start : 0);
      pool_cycles[1] = (end > start ? end - start : 0);
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif              
      // conv_3
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_3_params_alex8.batch_size, conv_3_params_alex8.in_dim, conv_3_params_alex8.in_channels,
          conv_3_params_alex8.out_channels, conv_3_params_alex8.out_dim,
          conv_3_params_alex8.stride, 1, conv_3_params_alex8.padding, conv_3_params_alex8.kernel_size,
          conv_3_params_alex8.out_stride,

          (elem_t*)conv_2_out_alex8_pooled, (elem_t*)conv_3_w_alex8, (acc_t*)conv_3_b_alex8, (elem_t*)conv_3_out_alex8,

          RELU, conv_3_params_alex8.output_scale, 0,
          conv_3_params_alex8.pool_size, 0, conv_3_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_4_params_alex8.batch_size, conv_4_params_alex8.in_dim, conv_4_params_alex8.in_channels,
          conv_4_params_alex8.out_channels, conv_4_params_alex8.out_dim,
          conv_4_params_alex8.stride, 1, conv_4_params_alex8.padding, conv_4_params_alex8.kernel_size,
          conv_4_params_alex8.out_stride,

          (elem_t*)conv_3_out_alex8, (elem_t*)conv_4_w_alex8, (acc_t*)conv_4_b_alex8, (elem_t*)conv_4_out_alex8,

          RELU, conv_4_params_alex8.output_scale, 0,
          conv_4_params_alex8.pool_size, 0, conv_4_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif        
          
      // conv_5
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_5_params_alex8.batch_size, conv_5_params_alex8.in_dim, conv_5_params_alex8.in_channels,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim,
          conv_5_params_alex8.stride, 1, conv_5_params_alex8.padding, conv_5_params_alex8.kernel_size,
          conv_5_params_alex8.out_stride,

          (elem_t*)conv_4_out_alex8, (elem_t*)conv_5_w_alex8, (acc_t*)conv_5_b_alex8, (elem_t*)conv_5_out_alex8,

          RELU, conv_5_params_alex8.output_scale, 0,
          1, 1, 0, false,
    //conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    if(part2){
	target_util = target_util_save;
      start = read_cycles();
      tiled_pool_auto_cid(
          conv_5_params_alex8.batch_size,
          conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, conv_5_params_alex8.out_dim_pooled,
          conv_5_params_alex8.out_stride,
          conv_5_params_alex8.pool_size, conv_5_params_alex8.pool_stride, conv_5_params_alex8.pool_padding,

          (elem_t*)conv_5_out_alex8, (elem_t*)conv_5_out_alex8_pooled,
    orow_divide, batch_divide, cid, group_id, target_util);
  gemmini_fence();
      end = read_cycles();
      total_pool_cycles += end - start;
      pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif           
      // Global averaging
      
      static elem_t average[8][9216] row_align(MAX_BLOCK_LEN);

      start = read_cycles();
      if(cid == 0)
          tiled_global_average_auto(conv_5_out_alex8_pooled, average, conv_5_params_alex8.batch_size,                         
              conv_5_params_alex8.out_channels, conv_5_params_alex8.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif
  int temp = batch_divide;
  batch_divide = orow_divide;
  orow_divide = temp;
      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8, (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8,
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      // fc_6
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_6_params_alex8.I, (int)(fc_6_params_alex8.J / 2), fc_6_params_alex8.K, fc_6_params_alex8.out_stride,
          (elem_t*)average, (elem_t*)fc_6_w_alex8 + (int)(fc_6_params_alex8.J / 2), (acc_t*)fc_6_b_alex8, (elem_t*)fc_6_out_alex8 + (int)(fc_6_params_alex8.J / 2),
          RELU, fc_6_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[0] += end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   


      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8, (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8,
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      // fc_7
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_7_params_alex8.I, (int)(fc_7_params_alex8.J / 2), fc_7_params_alex8.K, fc_7_params_alex8.out_stride,
          (elem_t*)fc_6_out_alex8, (elem_t*)fc_7_w_alex8 + (int)(fc_7_params_alex8.J / 2), (acc_t*)fc_7_b_alex8, (elem_t*)fc_7_out_alex8 + (int)(fc_7_params_alex8.J / 2),
          RELU, fc_7_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[1] += end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
      
      // fc_8
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_8_params_alex8.I, fc_8_params_alex8.J, fc_8_params_alex8.K, fc_8_params_alex8.out_stride,
          (elem_t*)fc_7_out_alex8, (elem_t*)fc_8_w_alex8, (acc_t*)fc_8_b_alex8, (elem_t*)fc_8_out_alex8,
          NO_ACTIVATION, fc_8_params_alex8.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_matmul_cycles += end - start;
      matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_alex);
#endif   
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 5){
        cycles[cid][i] = conv_cycles[i];
      }
      else if(i < 8){
        cycles[cid][i] = matmul_cycles[i - 5];
      }
      else if (i < 11){
        cycles[cid][i] = pool_cycles[i - 8];
      }
      else{
        if(i == 11) cycles[cid][i] = total_conv_cycles;
        if(i == 12) cycles[cid][i] = total_matmul_cycles;
        if(i == 13) cycles[cid][i] = total_pool_cycles;
        if(i == 14) cycles[cid][i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}
