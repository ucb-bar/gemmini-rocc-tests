
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet_params_4.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif


#ifndef BAREMETAL
uint64_t* resnet_function_4(size_t cid, size_t group_id, bool part1, bool part2, bool part3, bool part4, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_res){
#else
uint64_t* resnet_function_4(size_t cid, size_t group_id, bool part1, bool part2, bool part3, bool part4, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (20+34+16+3)
  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[54];//[20];
    //uint64_t conv_cycles[34];
    uint64_t resadd_cycles[16];
   //uint64_t target_cycle = target_cycles;
//printf("Address of start for cid %d: %p\n", cid, &start);
//printf("Address of end for cid %d: %p\n", cid, &end);
//printf("barrier: %d\n", barrier_res);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_res);
#endif

    if(part1){

      // conv_1
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_1_params_res4.batch_size, conv_1_params_res4.in_dim, conv_1_params_res4.in_channels,
          conv_1_params_res4.out_channels, conv_1_params_res4.out_dim,
          conv_1_params_res4.stride, 1, conv_1_params_res4.padding, conv_1_params_res4.kernel_size,
          conv_1_params_res4.out_stride,

          (elem_t*)image4_3, (elem_t*)conv_1_w_res4, (acc_t*)conv_1_b_res4, (elem_t*)conv_1_out_res4_pooled,

          RELU, conv_1_params_res4.output_scale, 0,
          conv_1_params_res4.pool_size, conv_1_params_res4.pool_stride, conv_1_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;

#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif             
      // conv_2
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_2_params_res4.I, conv_2_params_res4.J, conv_2_params_res4.K, conv_2_params_res4.out_stride,
          (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_2_w_res4, (acc_t*)conv_2_b_res4, (elem_t*)conv_2_out_res4,
          RELU, conv_2_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;

    #if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_3
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_3_params_res4.batch_size, conv_3_params_res4.in_dim, conv_3_params_res4.in_channels,
          conv_3_params_res4.out_channels, conv_3_params_res4.out_dim,
          conv_3_params_res4.stride, 1, conv_3_params_res4.padding, conv_3_params_res4.kernel_size,
          conv_3_params_res4.out_stride,

          (elem_t*)conv_2_out_res4, (elem_t*)conv_3_w_res4, (acc_t*)conv_3_b_res4, (elem_t*)conv_3_out_res4,

          RELU, conv_3_params_res4.output_scale, 0,
          conv_3_params_res4.pool_size, 0, conv_3_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;

#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_4_params_res4.I, conv_4_params_res4.J, conv_4_params_res4.K, conv_4_params_res4.out_stride,
          (elem_t*)conv_3_out_res4, (elem_t*)conv_4_w_res4, (acc_t*)conv_4_b_res4, (elem_t*)conv_4_out_res4,
          NO_ACTIVATION, conv_4_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_1_out_res4_pooled
      // conv_5
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_5_params_res4.I, conv_5_params_res4.J, conv_5_params_res4.K, conv_5_params_res4.out_stride,
          (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_5_w_res4, (acc_t*)conv_5_b_res4, (elem_t*)conv_5_out_res4,
          NO_ACTIVATION, conv_5_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_4_params_res4.I, conv_4_params_res4.J,
          conv_4_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_5_out_res4,
          (elem_t*)conv_4_out_res4,
          (elem_t*)conv_4_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_6
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_6_params_res4.I, conv_6_params_res4.J, conv_6_params_res4.K, conv_6_params_res4.out_stride,
          (elem_t*)conv_4_out_res4, (elem_t*)conv_6_w_res4, (acc_t*)conv_6_b_res4, (elem_t*)conv_6_out_res4,
          RELU, conv_6_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_7
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_7_params_res4.batch_size, conv_7_params_res4.in_dim, conv_7_params_res4.in_channels,
          conv_7_params_res4.out_channels, conv_7_params_res4.out_dim,
          conv_7_params_res4.stride, 1, conv_7_params_res4.padding, conv_7_params_res4.kernel_size,
          conv_7_params_res4.out_stride,

          (elem_t*)conv_6_out_res4, (elem_t*)conv_7_w_res4, (acc_t*)conv_7_b_res4, (elem_t*)conv_7_out_res4,

          RELU, conv_7_params_res4.output_scale, 0,
          conv_7_params_res4.pool_size, 0, conv_7_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_8
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_8_params_res4.I, conv_8_params_res4.J, conv_8_params_res4.K, conv_8_params_res4.out_stride,
          (elem_t*)conv_7_out_res4, (elem_t*)conv_8_w_res4, (acc_t*)conv_8_b_res4, (elem_t*)conv_8_out_res4,
          NO_ACTIVATION, conv_8_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_8_params_res4.I, conv_8_params_res4.J,
          conv_8_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_4_out_res4,
          (elem_t*)conv_8_out_res4,
          (elem_t*)conv_8_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_9
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_9_params_res4.I, conv_9_params_res4.J, conv_9_params_res4.K, conv_9_params_res4.out_stride,
          (elem_t*)conv_8_out_res4, (elem_t*)conv_9_w_res4, (acc_t*)conv_9_b_res4, (elem_t*)conv_9_out_res4,
          RELU, conv_9_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_10
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_10_params_res4.batch_size, conv_10_params_res4.in_dim, conv_10_params_res4.in_channels,
          conv_10_params_res4.out_channels, conv_10_params_res4.out_dim,
          conv_10_params_res4.stride, 1, conv_10_params_res4.padding, conv_10_params_res4.kernel_size,
          conv_10_params_res4.out_stride,

          (elem_t*)conv_9_out_res4, (elem_t*)conv_10_w_res4, (acc_t*)conv_10_b_res4, (elem_t*)conv_10_out_res4,

          RELU, conv_10_params_res4.output_scale, 0,
          conv_10_params_res4.pool_size, 0, conv_10_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_11
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_11_params_res4.I, conv_11_params_res4.J, conv_11_params_res4.K, conv_11_params_res4.out_stride,
          (elem_t*)conv_10_out_res4, (elem_t*)conv_11_w_res4, (acc_t*)conv_11_b_res4, (elem_t*)conv_11_out_res4,
          NO_ACTIVATION, conv_11_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_11_params_res4.I, conv_11_params_res4.J,
          conv_11_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_8_out_res4,
          (elem_t*)conv_11_out_res4,
          (elem_t*)conv_11_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_12
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_12_params_res4.I, conv_12_params_res4.J, conv_12_params_res4.K, conv_12_params_res4.out_stride,
          (elem_t*)conv_11_out_res4, (elem_t*)conv_12_w_res4, (acc_t*)conv_12_b_res4, (elem_t*)conv_12_out_res4,
          RELU, conv_12_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
    }

    if(part2){
      // conv_13
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_13_params_res4.batch_size, conv_13_params_res4.in_dim, conv_13_params_res4.in_channels,
          conv_13_params_res4.out_channels, conv_13_params_res4.out_dim,
          conv_13_params_res4.stride, 1, conv_13_params_res4.padding, conv_13_params_res4.kernel_size,
          conv_13_params_res4.out_stride,

          (elem_t*)conv_12_out_res4, (elem_t*)conv_13_w_res4, (acc_t*)conv_13_b_res4, (elem_t*)conv_13_out_res4,

          RELU, conv_13_params_res4.output_scale, 0,
          conv_13_params_res4.pool_size, 0, conv_13_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_14
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_14_params_res4.I, conv_14_params_res4.J, conv_14_params_res4.K, conv_14_params_res4.out_stride,
          (elem_t*)conv_13_out_res4, (elem_t*)conv_14_w_res4, (acc_t*)conv_14_b_res4, (elem_t*)conv_14_out_res4,
          NO_ACTIVATION, conv_14_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_11_out_res4
      // conv_15
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_15_params_res4.batch_size, conv_15_params_res4.in_dim, conv_15_params_res4.in_channels,
          conv_15_params_res4.out_channels, conv_15_params_res4.out_dim,
          conv_15_params_res4.stride, 1, conv_15_params_res4.padding, conv_15_params_res4.kernel_size,
          conv_15_params_res4.out_stride,

          (elem_t*)conv_11_out_res4, (elem_t*)conv_15_w_res4, (acc_t*)conv_15_b_res4, (elem_t*)conv_15_out_res4,

          NO_ACTIVATION, conv_15_params_res4.output_scale, 0,
          conv_15_params_res4.pool_size, 0, conv_15_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_14_params_res4.I, conv_14_params_res4.J,
          conv_14_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_15_out_res4,
          (elem_t*)conv_14_out_res4,
          (elem_t*)conv_14_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_16
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_16_params_res4.I, conv_16_params_res4.J, conv_16_params_res4.K, conv_16_params_res4.out_stride,
          (elem_t*)conv_14_out_res4, (elem_t*)conv_16_w_res4, (acc_t*)conv_16_b_res4, (elem_t*)conv_16_out_res4,
          RELU, conv_16_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_17
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_17_params_res4.batch_size, conv_17_params_res4.in_dim, conv_17_params_res4.in_channels,
          conv_17_params_res4.out_channels, conv_17_params_res4.out_dim,
          conv_17_params_res4.stride, 1, conv_17_params_res4.padding, conv_17_params_res4.kernel_size,
          conv_17_params_res4.out_stride,

          (elem_t*)conv_16_out_res4, (elem_t*)conv_17_w_res4, (acc_t*)conv_17_b_res4, (elem_t*)conv_17_out_res4,

          RELU, conv_17_params_res4.output_scale, 0,
          conv_17_params_res4.pool_size, 0, conv_17_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_18
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_18_params_res4.I, conv_18_params_res4.J, conv_18_params_res4.K, conv_18_params_res4.out_stride,
          (elem_t*)conv_17_out_res4, (elem_t*)conv_18_w_res4, (acc_t*)conv_18_b_res4, (elem_t*)conv_18_out_res4,
          NO_ACTIVATION, conv_18_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_18_params_res4.I, conv_18_params_res4.J,
          conv_18_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_14_out_res4,
          (elem_t*)conv_18_out_res4,
          (elem_t*)conv_18_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_19
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_19_params_res4.I, conv_19_params_res4.J, conv_19_params_res4.K, conv_19_params_res4.out_stride,
          (elem_t*)conv_18_out_res4, (elem_t*)conv_19_w_res4, (acc_t*)conv_19_b_res4, (elem_t*)conv_19_out_res4,
          RELU, conv_19_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_20
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_20_params_res4.batch_size, conv_20_params_res4.in_dim, conv_20_params_res4.in_channels,
          conv_20_params_res4.out_channels, conv_20_params_res4.out_dim,
          conv_20_params_res4.stride, 1, conv_20_params_res4.padding, conv_20_params_res4.kernel_size,
          conv_20_params_res4.out_stride,

          (elem_t*)conv_19_out_res4, (elem_t*)conv_20_w_res4, (acc_t*)conv_20_b_res4, (elem_t*)conv_20_out_res4,

          RELU, conv_20_params_res4.output_scale, 0,
          conv_20_params_res4.pool_size, 0, conv_20_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_21
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_21_params_res4.I, conv_21_params_res4.J, conv_21_params_res4.K, conv_21_params_res4.out_stride,
          (elem_t*)conv_20_out_res4, (elem_t*)conv_21_w_res4, (acc_t*)conv_21_b_res4, (elem_t*)conv_21_out_res4,
          NO_ACTIVATION, conv_21_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_21_params_res4.I, conv_21_params_res4.J,
          conv_21_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_18_out_res4,
          (elem_t*)conv_21_out_res4,
          (elem_t*)conv_21_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_22
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_22_params_res4.I, conv_22_params_res4.J, conv_22_params_res4.K, conv_22_params_res4.out_stride,
          (elem_t*)conv_21_out_res4, (elem_t*)conv_22_w_res4, (acc_t*)conv_22_b_res4, (elem_t*)conv_22_out_res4,
          RELU, conv_22_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_23
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_23_params_res4.batch_size, conv_23_params_res4.in_dim, conv_23_params_res4.in_channels,
          conv_23_params_res4.out_channels, conv_23_params_res4.out_dim,
          conv_23_params_res4.stride, 1, conv_23_params_res4.padding, conv_23_params_res4.kernel_size,
          conv_23_params_res4.out_stride,

          (elem_t*)conv_22_out_res4, (elem_t*)conv_23_w_res4, (acc_t*)conv_23_b_res4, (elem_t*)conv_23_out_res4,

          RELU, conv_23_params_res4.output_scale, 0,
          conv_23_params_res4.pool_size, 0, conv_23_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_24
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_24_params_res4.I, conv_24_params_res4.J, conv_24_params_res4.K, conv_24_params_res4.out_stride,
          (elem_t*)conv_23_out_res4, (elem_t*)conv_24_w_res4, (acc_t*)conv_24_b_res4, (elem_t*)conv_24_out_res4,
          NO_ACTIVATION, conv_24_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_24_params_res4.I, conv_24_params_res4.J,
          conv_24_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_21_out_res4,
          (elem_t*)conv_24_out_res4,
          (elem_t*)conv_24_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_25
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_25_params_res4.I, conv_25_params_res4.J, conv_25_params_res4.K, conv_25_params_res4.out_stride,
          (elem_t*)conv_24_out_res4, (elem_t*)conv_25_w_res4, (acc_t*)conv_25_b_res4, (elem_t*)conv_25_out_res4,
          RELU, conv_25_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
    }

    if(part3){
      // conv_26
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_26_params_res4.batch_size, conv_26_params_res4.in_dim, conv_26_params_res4.in_channels,
          conv_26_params_res4.out_channels, conv_26_params_res4.out_dim,
          conv_26_params_res4.stride, 1, conv_26_params_res4.padding, conv_26_params_res4.kernel_size,
          conv_26_params_res4.out_stride,

          (elem_t*)conv_25_out_res4, (elem_t*)conv_26_w_res4, (acc_t*)conv_26_b_res4, (elem_t*)conv_26_out_res4,

          RELU, conv_26_params_res4.output_scale, 0,
          conv_26_params_res4.pool_size, 0, conv_26_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_27
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_27_params_res4.I, conv_27_params_res4.J, conv_27_params_res4.K, conv_27_params_res4.out_stride,
          (elem_t*)conv_26_out_res4, (elem_t*)conv_27_w_res4, (acc_t*)conv_27_b_res4, (elem_t*)conv_27_out_res4,
          NO_ACTIVATION, conv_27_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_24_out_res4
      // conv_28
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_28_params_res4.batch_size, conv_28_params_res4.in_dim, conv_28_params_res4.in_channels,
          conv_28_params_res4.out_channels, conv_28_params_res4.out_dim,
          conv_28_params_res4.stride, 1, conv_28_params_res4.padding, conv_28_params_res4.kernel_size,
          conv_28_params_res4.out_stride,

          (elem_t*)conv_24_out_res4, (elem_t*)conv_28_w_res4, (acc_t*)conv_28_b_res4, (elem_t*)conv_28_out_res4,

          NO_ACTIVATION, conv_28_params_res4.output_scale, 0,
          conv_28_params_res4.pool_size, 0, conv_28_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_27_params_res4.I, conv_27_params_res4.J,
          conv_27_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_28_out_res4,
          (elem_t*)conv_27_out_res4,
          (elem_t*)conv_27_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_29
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_29_params_res4.I, conv_29_params_res4.J, conv_29_params_res4.K, conv_29_params_res4.out_stride,
          (elem_t*)conv_27_out_res4, (elem_t*)conv_29_w_res4, (acc_t*)conv_29_b_res4, (elem_t*)conv_29_out_res4,
          RELU, conv_29_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_30
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_30_params_res4.batch_size, conv_30_params_res4.in_dim, conv_30_params_res4.in_channels,
          conv_30_params_res4.out_channels, conv_30_params_res4.out_dim,
          conv_30_params_res4.stride, 1, conv_30_params_res4.padding, conv_30_params_res4.kernel_size,
          conv_30_params_res4.out_stride,

          (elem_t*)conv_29_out_res4, (elem_t*)conv_30_w_res4, (acc_t*)conv_30_b_res4, (elem_t*)conv_30_out_res4,

          RELU, conv_30_params_res4.output_scale, 0,
          conv_30_params_res4.pool_size, 0, conv_30_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_31
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_31_params_res4.I, conv_31_params_res4.J, conv_31_params_res4.K, conv_31_params_res4.out_stride,
          (elem_t*)conv_30_out_res4, (elem_t*)conv_31_w_res4, (acc_t*)conv_31_b_res4, (elem_t*)conv_31_out_res4,
          NO_ACTIVATION, conv_31_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_31_params_res4.I, conv_31_params_res4.J,
          conv_31_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_27_out_res4,
          (elem_t*)conv_31_out_res4,
          (elem_t*)conv_31_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[8] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_32
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_32_params_res4.I, conv_32_params_res4.J, conv_32_params_res4.K, conv_32_params_res4.out_stride,
          (elem_t*)conv_31_out_res4, (elem_t*)conv_32_w_res4, (acc_t*)conv_32_b_res4, (elem_t*)conv_32_out_res4,
          RELU, conv_32_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_33
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_33_params_res4.batch_size, conv_33_params_res4.in_dim, conv_33_params_res4.in_channels,
          conv_33_params_res4.out_channels, conv_33_params_res4.out_dim,
          conv_33_params_res4.stride, 1, conv_33_params_res4.padding, conv_33_params_res4.kernel_size,
          conv_33_params_res4.out_stride,

          (elem_t*)conv_32_out_res4, (elem_t*)conv_33_w_res4, (acc_t*)conv_33_b_res4, (elem_t*)conv_33_out_res4,

          RELU, conv_33_params_res4.output_scale, 0,
          conv_33_params_res4.pool_size, 0, conv_33_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_34
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_34_params_res4.I, conv_34_params_res4.J, conv_34_params_res4.K, conv_34_params_res4.out_stride,
          (elem_t*)conv_33_out_res4, (elem_t*)conv_34_w_res4, (acc_t*)conv_34_b_res4, (elem_t*)conv_34_out_res4,
          NO_ACTIVATION, conv_34_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_34_params_res4.I, conv_34_params_res4.J,
          conv_34_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_31_out_res4,
          (elem_t*)conv_34_out_res4,
          (elem_t*)conv_34_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[9] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_35
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_35_params_res4.I, conv_35_params_res4.J, conv_35_params_res4.K, conv_35_params_res4.out_stride,
          (elem_t*)conv_34_out_res4, (elem_t*)conv_35_w_res4, (acc_t*)conv_35_b_res4, (elem_t*)conv_35_out_res4,
          RELU, conv_35_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_36
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_36_params_res4.batch_size, conv_36_params_res4.in_dim, conv_36_params_res4.in_channels,
          conv_36_params_res4.out_channels, conv_36_params_res4.out_dim,
          conv_36_params_res4.stride, 1, conv_36_params_res4.padding, conv_36_params_res4.kernel_size,
          conv_36_params_res4.out_stride,

          (elem_t*)conv_35_out_res4, (elem_t*)conv_36_w_res4, (acc_t*)conv_36_b_res4, (elem_t*)conv_36_out_res4,

          RELU, conv_36_params_res4.output_scale, 0,
          conv_36_params_res4.pool_size, 0, conv_36_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_37
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_37_params_res4.I, conv_37_params_res4.J, conv_37_params_res4.K, conv_37_params_res4.out_stride,
          (elem_t*)conv_36_out_res4, (elem_t*)conv_37_w_res4, (acc_t*)conv_37_b_res4, (elem_t*)conv_37_out_res4,
          NO_ACTIVATION, conv_37_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_37_params_res4.I, conv_37_params_res4.J,
          conv_37_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_34_out_res4,
          (elem_t*)conv_37_out_res4,
          (elem_t*)conv_37_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[10] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_38
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_38_params_res4.I, conv_38_params_res4.J, conv_38_params_res4.K, conv_38_params_res4.out_stride,
          (elem_t*)conv_37_out_res4, (elem_t*)conv_38_w_res4, (acc_t*)conv_38_b_res4, (elem_t*)conv_38_out_res4,
          RELU, conv_38_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_39
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_39_params_res4.batch_size, conv_39_params_res4.in_dim, conv_39_params_res4.in_channels,
          conv_39_params_res4.out_channels, conv_39_params_res4.out_dim,
          conv_39_params_res4.stride, 1, conv_39_params_res4.padding, conv_39_params_res4.kernel_size,
          conv_39_params_res4.out_stride,

          (elem_t*)conv_38_out_res4, (elem_t*)conv_39_w_res4, (acc_t*)conv_39_b_res4, (elem_t*)conv_39_out_res4,

          RELU, conv_39_params_res4.output_scale, 0,
          conv_39_params_res4.pool_size, 0, conv_39_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_40
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_40_params_res4.I, conv_40_params_res4.J, conv_40_params_res4.K, conv_40_params_res4.out_stride,
          (elem_t*)conv_39_out_res4, (elem_t*)conv_40_w_res4, (acc_t*)conv_40_b_res4, (elem_t*)conv_40_out_res4,
          NO_ACTIVATION, conv_40_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_40_params_res4.I, conv_40_params_res4.J,
          conv_40_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_37_out_res4,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_40_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[11] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_41
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_41_params_res4.I, conv_41_params_res4.J, conv_41_params_res4.K, conv_41_params_res4.out_stride,
          (elem_t*)conv_40_out_res4, (elem_t*)conv_41_w_res4, (acc_t*)conv_41_b_res4, (elem_t*)conv_41_out_res4,
          RELU, conv_41_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_42
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_42_params_res4.batch_size, conv_42_params_res4.in_dim, conv_42_params_res4.in_channels,
          conv_42_params_res4.out_channels, conv_42_params_res4.out_dim,
          conv_42_params_res4.stride, 1, conv_42_params_res4.padding, conv_42_params_res4.kernel_size,
          conv_42_params_res4.out_stride,

          (elem_t*)conv_41_out_res4, (elem_t*)conv_42_w_res4, (acc_t*)conv_42_b_res4, (elem_t*)conv_42_out_res4,

          RELU, conv_42_params_res4.output_scale, 0,
          conv_42_params_res4.pool_size, 0, conv_42_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_43
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_43_params_res4.I, conv_43_params_res4.J, conv_43_params_res4.K, conv_43_params_res4.out_stride,
          (elem_t*)conv_42_out_res4, (elem_t*)conv_43_w_res4, (acc_t*)conv_43_b_res4, (elem_t*)conv_43_out_res4,
          NO_ACTIVATION, conv_43_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_43_params_res4.I, conv_43_params_res4.J,
          conv_43_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_43_out_res4,
          (elem_t*)conv_43_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[12] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
    
        // conv_44
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_44_params_res4.I, conv_44_params_res4.J, conv_44_params_res4.K, conv_44_params_res4.out_stride,
            (elem_t*)conv_43_out_res4, (elem_t*)conv_44_w_res4, (acc_t*)conv_44_b_res4, (elem_t*)conv_44_out_res4,
            RELU, conv_44_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
    }

    if(part4){
      // conv_45
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_45_params_res4.batch_size, conv_45_params_res4.in_dim, conv_45_params_res4.in_channels,
          conv_45_params_res4.out_channels, conv_45_params_res4.out_dim,
          conv_45_params_res4.stride, 1, conv_45_params_res4.padding, conv_45_params_res4.kernel_size,
          conv_45_params_res4.out_stride,

          (elem_t*)conv_44_out_res4, (elem_t*)conv_45_w_res4, (acc_t*)conv_45_b_res4, (elem_t*)conv_45_out_res4,

          RELU, conv_45_params_res4.output_scale, 0,
          conv_45_params_res4.pool_size, 0, conv_45_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_46
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_46_params_res4.I, conv_46_params_res4.J, conv_46_params_res4.K, conv_46_params_res4.out_stride,
          (elem_t*)conv_45_out_res4, (elem_t*)conv_46_w_res4, (acc_t*)conv_46_b_res4, (elem_t*)conv_46_out_res4,
          NO_ACTIVATION, conv_46_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_43_out_res4
      // conv_47
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_47_params_res4.batch_size, conv_47_params_res4.in_dim, conv_47_params_res4.in_channels,
          conv_47_params_res4.out_channels, conv_47_params_res4.out_dim,
          conv_47_params_res4.stride, 1, conv_47_params_res4.padding, conv_47_params_res4.kernel_size,
          conv_47_params_res4.out_stride,

          (elem_t*)conv_43_out_res4, (elem_t*)conv_47_w_res4, (acc_t*)conv_47_b_res4, (elem_t*)conv_47_out_res4,

          NO_ACTIVATION, conv_47_params_res4.output_scale, 0,
          conv_47_params_res4.pool_size, 0, conv_47_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_46_params_res4.I, conv_46_params_res4.J,
          conv_46_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_47_out_res4,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_46_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[13] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_48
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_48_params_res4.I, conv_48_params_res4.J, conv_48_params_res4.K, conv_48_params_res4.out_stride,
          (elem_t*)conv_46_out_res4, (elem_t*)conv_48_w_res4, (acc_t*)conv_48_b_res4, (elem_t*)conv_48_out_res4,
          RELU, conv_48_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_49
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_49_params_res4.batch_size, conv_49_params_res4.in_dim, conv_49_params_res4.in_channels,
          conv_49_params_res4.out_channels, conv_49_params_res4.out_dim,
          conv_49_params_res4.stride, 1, conv_49_params_res4.padding, conv_49_params_res4.kernel_size,
          conv_49_params_res4.out_stride,

          (elem_t*)conv_48_out_res4, (elem_t*)conv_49_w_res4, (acc_t*)conv_49_b_res4, (elem_t*)conv_49_out_res4,

          RELU, conv_49_params_res4.output_scale, 0,
          conv_49_params_res4.pool_size, 0, conv_49_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_50
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_50_params_res4.I, conv_50_params_res4.J, conv_50_params_res4.K, conv_50_params_res4.out_stride,
          (elem_t*)conv_49_out_res4, (elem_t*)conv_50_w_res4, (acc_t*)conv_50_b_res4, (elem_t*)conv_50_out_res4,
          NO_ACTIVATION, conv_50_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_50_params_res4.I, conv_50_params_res4.J,
          conv_50_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_50_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[14] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_51
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_51_params_res4.I, conv_51_params_res4.J, conv_51_params_res4.K, conv_51_params_res4.out_stride,
          (elem_t*)conv_50_out_res4, (elem_t*)conv_51_w_res4, (acc_t*)conv_51_b_res4, (elem_t*)conv_51_out_res4,
          RELU, conv_51_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_52
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_52_params_res4.batch_size, conv_52_params_res4.in_dim, conv_52_params_res4.in_channels,
          conv_52_params_res4.out_channels, conv_52_params_res4.out_dim,
          conv_52_params_res4.stride, 1, conv_52_params_res4.padding, conv_52_params_res4.kernel_size,
          conv_52_params_res4.out_stride,

          (elem_t*)conv_51_out_res4, (elem_t*)conv_52_w_res4, (acc_t*)conv_52_b_res4, (elem_t*)conv_52_out_res4,

          RELU, conv_52_params_res4.output_scale, 0,
          conv_52_params_res4.pool_size, 0, conv_52_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_53
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_53_params_res4.I, conv_53_params_res4.J, conv_53_params_res4.K, conv_53_params_res4.out_stride,
          (elem_t*)conv_52_out_res4, (elem_t*)conv_53_w_res4, (acc_t*)conv_53_b_res4, (elem_t*)conv_53_out_res4,
          NO_ACTIVATION, conv_53_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_53_params_res4.I, conv_53_params_res4.J,
          conv_53_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_53_out_res4,
          (elem_t*)conv_53_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[15] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
      
      // Global averaging
      
      static elem_t average[4][2048] row_align(MAX_BLOCK_LEN);

      start = read_cycles();
      if(cid == 1)
          tiled_global_average_auto(conv_53_out_res4, average, conv_53_params_res4.batch_size,                         
              conv_53_params_res4.out_channels, conv_53_params_res4.out_dim, WS);
      else
	gemmini_dram_util[group_id] = 0;    

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
	target_util = -1; // for last layer  
      // fc_54
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_54_params_res4.I, fc_54_params_res4.J, fc_54_params_res4.K, fc_54_params_res4.out_stride,
          (elem_t*)average, (elem_t*)fc_54_w_res4, (acc_t*)fc_54_b_res4, (elem_t*)fc_54_out_res4,
          NO_ACTIVATION, fc_54_params_res4.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif  
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 54){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 70){
        cycles[cid][i] = resadd_cycles[i - 54];
      }
      else{
        if(i == 70) cycles[cid][i] = total_conv_cycles;
        if(i == 71) cycles[cid][i] = total_resadd_cycles;
        if(i == 72) cycles[cid][i] = total_conv_cycles + total_resadd_cycles + other_cycles;
      }
    }

    return cycles[cid];
#undef num_cycle
}

uint64_t* resnet_block_function_4(size_t cid, size_t group_id, bool part1, bool part2, bool part3, bool part4, int orow_divide, int batch_divide, int target_util){
#define num_cycle (20+34+16+3)
  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[54];//[20];
    //uint64_t conv_cycles[34];
    uint64_t resadd_cycles[16];
   //uint64_t target_cycle = target_cycles;
//printf("Address of start for cid %d: %p\n", cid, &start);
//printf("Address of end for cid %d: %p\n", cid, &end);
//printf("barrier: %d\n", barrier_res);
#if THREAD_SYNC == 1
    //pthread_barrier_wait(barrier_res);
#endif

    if(part1){

      // conv_1
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_1_params_res4.batch_size, conv_1_params_res4.in_dim, conv_1_params_res4.in_channels,
          conv_1_params_res4.out_channels, conv_1_params_res4.out_dim,
          conv_1_params_res4.stride, 1, conv_1_params_res4.padding, conv_1_params_res4.kernel_size,
          conv_1_params_res4.out_stride,

          (elem_t*)image4_3, (elem_t*)conv_1_w_res4, (acc_t*)conv_1_b_res4, (elem_t*)conv_1_out_res4_pooled,

          RELU, conv_1_params_res4.output_scale, 0,
          conv_1_params_res4.pool_size, conv_1_params_res4.pool_stride, conv_1_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[0] = end - start;

#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif             
      // conv_2
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_2_params_res4.I, conv_2_params_res4.J, conv_2_params_res4.K, conv_2_params_res4.out_stride,
          (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_2_w_res4, (acc_t*)conv_2_b_res4, (elem_t*)conv_2_out_res4,
          RELU, conv_2_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[1] = end - start;

    #if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_3
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_3_params_res4.batch_size, conv_3_params_res4.in_dim, conv_3_params_res4.in_channels,
          conv_3_params_res4.out_channels, conv_3_params_res4.out_dim,
          conv_3_params_res4.stride, 1, conv_3_params_res4.padding, conv_3_params_res4.kernel_size,
          conv_3_params_res4.out_stride,

          (elem_t*)conv_2_out_res4, (elem_t*)conv_3_w_res4, (acc_t*)conv_3_b_res4, (elem_t*)conv_3_out_res4,

          RELU, conv_3_params_res4.output_scale, 0,
          conv_3_params_res4.pool_size, 0, conv_3_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[2] = end - start;

#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_4
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_4_params_res4.I, conv_4_params_res4.J, conv_4_params_res4.K, conv_4_params_res4.out_stride,
          (elem_t*)conv_3_out_res4, (elem_t*)conv_4_w_res4, (acc_t*)conv_4_b_res4, (elem_t*)conv_4_out_res4,
          NO_ACTIVATION, conv_4_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_1_out_res4_pooled
      // conv_5
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_5_params_res4.I, conv_5_params_res4.J, conv_5_params_res4.K, conv_5_params_res4.out_stride,
          (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_5_w_res4, (acc_t*)conv_5_b_res4, (elem_t*)conv_5_out_res4,
          NO_ACTIVATION, conv_5_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_4_params_res4.I, conv_4_params_res4.J,
          conv_4_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_5_out_res4,
          (elem_t*)conv_4_out_res4,
          (elem_t*)conv_4_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_6
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_6_params_res4.I, conv_6_params_res4.J, conv_6_params_res4.K, conv_6_params_res4.out_stride,
          (elem_t*)conv_4_out_res4, (elem_t*)conv_6_w_res4, (acc_t*)conv_6_b_res4, (elem_t*)conv_6_out_res4,
          RELU, conv_6_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_7
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_7_params_res4.batch_size, conv_7_params_res4.in_dim, conv_7_params_res4.in_channels,
          conv_7_params_res4.out_channels, conv_7_params_res4.out_dim,
          conv_7_params_res4.stride, 1, conv_7_params_res4.padding, conv_7_params_res4.kernel_size,
          conv_7_params_res4.out_stride,

          (elem_t*)conv_6_out_res4, (elem_t*)conv_7_w_res4, (acc_t*)conv_7_b_res4, (elem_t*)conv_7_out_res4,

          RELU, conv_7_params_res4.output_scale, 0,
          conv_7_params_res4.pool_size, 0, conv_7_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_8
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_8_params_res4.I, conv_8_params_res4.J, conv_8_params_res4.K, conv_8_params_res4.out_stride,
          (elem_t*)conv_7_out_res4, (elem_t*)conv_8_w_res4, (acc_t*)conv_8_b_res4, (elem_t*)conv_8_out_res4,
          NO_ACTIVATION, conv_8_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_8_params_res4.I, conv_8_params_res4.J,
          conv_8_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_4_out_res4,
          (elem_t*)conv_8_out_res4,
          (elem_t*)conv_8_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_9
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_9_params_res4.I, conv_9_params_res4.J, conv_9_params_res4.K, conv_9_params_res4.out_stride,
          (elem_t*)conv_8_out_res4, (elem_t*)conv_9_w_res4, (acc_t*)conv_9_b_res4, (elem_t*)conv_9_out_res4,
          RELU, conv_9_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_10
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_10_params_res4.batch_size, conv_10_params_res4.in_dim, conv_10_params_res4.in_channels,
          conv_10_params_res4.out_channels, conv_10_params_res4.out_dim,
          conv_10_params_res4.stride, 1, conv_10_params_res4.padding, conv_10_params_res4.kernel_size,
          conv_10_params_res4.out_stride,

          (elem_t*)conv_9_out_res4, (elem_t*)conv_10_w_res4, (acc_t*)conv_10_b_res4, (elem_t*)conv_10_out_res4,

          RELU, conv_10_params_res4.output_scale, 0,
          conv_10_params_res4.pool_size, 0, conv_10_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_11
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_11_params_res4.I, conv_11_params_res4.J, conv_11_params_res4.K, conv_11_params_res4.out_stride,
          (elem_t*)conv_10_out_res4, (elem_t*)conv_11_w_res4, (acc_t*)conv_11_b_res4, (elem_t*)conv_11_out_res4,
          NO_ACTIVATION, conv_11_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_11_params_res4.I, conv_11_params_res4.J,
          conv_11_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_8_out_res4,
          (elem_t*)conv_11_out_res4,
          (elem_t*)conv_11_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_12
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_12_params_res4.I, conv_12_params_res4.J, conv_12_params_res4.K, conv_12_params_res4.out_stride,
          (elem_t*)conv_11_out_res4, (elem_t*)conv_12_w_res4, (acc_t*)conv_12_b_res4, (elem_t*)conv_12_out_res4,
          RELU, conv_12_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
    }

    if(part2){
      // conv_13
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_13_params_res4.batch_size, conv_13_params_res4.in_dim, conv_13_params_res4.in_channels,
          conv_13_params_res4.out_channels, conv_13_params_res4.out_dim,
          conv_13_params_res4.stride, 1, conv_13_params_res4.padding, conv_13_params_res4.kernel_size,
          conv_13_params_res4.out_stride,

          (elem_t*)conv_12_out_res4, (elem_t*)conv_13_w_res4, (acc_t*)conv_13_b_res4, (elem_t*)conv_13_out_res4,

          RELU, conv_13_params_res4.output_scale, 0,
          conv_13_params_res4.pool_size, 0, conv_13_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_14
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_14_params_res4.I, conv_14_params_res4.J, conv_14_params_res4.K, conv_14_params_res4.out_stride,
          (elem_t*)conv_13_out_res4, (elem_t*)conv_14_w_res4, (acc_t*)conv_14_b_res4, (elem_t*)conv_14_out_res4,
          NO_ACTIVATION, conv_14_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_11_out_res4
      // conv_15
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_15_params_res4.batch_size, conv_15_params_res4.in_dim, conv_15_params_res4.in_channels,
          conv_15_params_res4.out_channels, conv_15_params_res4.out_dim,
          conv_15_params_res4.stride, 1, conv_15_params_res4.padding, conv_15_params_res4.kernel_size,
          conv_15_params_res4.out_stride,

          (elem_t*)conv_11_out_res4, (elem_t*)conv_15_w_res4, (acc_t*)conv_15_b_res4, (elem_t*)conv_15_out_res4,

          NO_ACTIVATION, conv_15_params_res4.output_scale, 0,
          conv_15_params_res4.pool_size, 0, conv_15_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_14_params_res4.I, conv_14_params_res4.J,
          conv_14_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_15_out_res4,
          (elem_t*)conv_14_out_res4,
          (elem_t*)conv_14_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_16
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_16_params_res4.I, conv_16_params_res4.J, conv_16_params_res4.K, conv_16_params_res4.out_stride,
          (elem_t*)conv_14_out_res4, (elem_t*)conv_16_w_res4, (acc_t*)conv_16_b_res4, (elem_t*)conv_16_out_res4,
          RELU, conv_16_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_17
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_17_params_res4.batch_size, conv_17_params_res4.in_dim, conv_17_params_res4.in_channels,
          conv_17_params_res4.out_channels, conv_17_params_res4.out_dim,
          conv_17_params_res4.stride, 1, conv_17_params_res4.padding, conv_17_params_res4.kernel_size,
          conv_17_params_res4.out_stride,

          (elem_t*)conv_16_out_res4, (elem_t*)conv_17_w_res4, (acc_t*)conv_17_b_res4, (elem_t*)conv_17_out_res4,

          RELU, conv_17_params_res4.output_scale, 0,
          conv_17_params_res4.pool_size, 0, conv_17_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_18
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_18_params_res4.I, conv_18_params_res4.J, conv_18_params_res4.K, conv_18_params_res4.out_stride,
          (elem_t*)conv_17_out_res4, (elem_t*)conv_18_w_res4, (acc_t*)conv_18_b_res4, (elem_t*)conv_18_out_res4,
          NO_ACTIVATION, conv_18_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_18_params_res4.I, conv_18_params_res4.J,
          conv_18_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_14_out_res4,
          (elem_t*)conv_18_out_res4,
          (elem_t*)conv_18_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_19
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_19_params_res4.I, conv_19_params_res4.J, conv_19_params_res4.K, conv_19_params_res4.out_stride,
          (elem_t*)conv_18_out_res4, (elem_t*)conv_19_w_res4, (acc_t*)conv_19_b_res4, (elem_t*)conv_19_out_res4,
          RELU, conv_19_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_20
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_20_params_res4.batch_size, conv_20_params_res4.in_dim, conv_20_params_res4.in_channels,
          conv_20_params_res4.out_channels, conv_20_params_res4.out_dim,
          conv_20_params_res4.stride, 1, conv_20_params_res4.padding, conv_20_params_res4.kernel_size,
          conv_20_params_res4.out_stride,

          (elem_t*)conv_19_out_res4, (elem_t*)conv_20_w_res4, (acc_t*)conv_20_b_res4, (elem_t*)conv_20_out_res4,

          RELU, conv_20_params_res4.output_scale, 0,
          conv_20_params_res4.pool_size, 0, conv_20_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_21
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_21_params_res4.I, conv_21_params_res4.J, conv_21_params_res4.K, conv_21_params_res4.out_stride,
          (elem_t*)conv_20_out_res4, (elem_t*)conv_21_w_res4, (acc_t*)conv_21_b_res4, (elem_t*)conv_21_out_res4,
          NO_ACTIVATION, conv_21_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_21_params_res4.I, conv_21_params_res4.J,
          conv_21_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_18_out_res4,
          (elem_t*)conv_21_out_res4,
          (elem_t*)conv_21_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_22
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_22_params_res4.I, conv_22_params_res4.J, conv_22_params_res4.K, conv_22_params_res4.out_stride,
          (elem_t*)conv_21_out_res4, (elem_t*)conv_22_w_res4, (acc_t*)conv_22_b_res4, (elem_t*)conv_22_out_res4,
          RELU, conv_22_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_23
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_23_params_res4.batch_size, conv_23_params_res4.in_dim, conv_23_params_res4.in_channels,
          conv_23_params_res4.out_channels, conv_23_params_res4.out_dim,
          conv_23_params_res4.stride, 1, conv_23_params_res4.padding, conv_23_params_res4.kernel_size,
          conv_23_params_res4.out_stride,

          (elem_t*)conv_22_out_res4, (elem_t*)conv_23_w_res4, (acc_t*)conv_23_b_res4, (elem_t*)conv_23_out_res4,

          RELU, conv_23_params_res4.output_scale, 0,
          conv_23_params_res4.pool_size, 0, conv_23_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_24
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_24_params_res4.I, conv_24_params_res4.J, conv_24_params_res4.K, conv_24_params_res4.out_stride,
          (elem_t*)conv_23_out_res4, (elem_t*)conv_24_w_res4, (acc_t*)conv_24_b_res4, (elem_t*)conv_24_out_res4,
          NO_ACTIVATION, conv_24_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_24_params_res4.I, conv_24_params_res4.J,
          conv_24_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_21_out_res4,
          (elem_t*)conv_24_out_res4,
          (elem_t*)conv_24_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_25
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_25_params_res4.I, conv_25_params_res4.J, conv_25_params_res4.K, conv_25_params_res4.out_stride,
          (elem_t*)conv_24_out_res4, (elem_t*)conv_25_w_res4, (acc_t*)conv_25_b_res4, (elem_t*)conv_25_out_res4,
          RELU, conv_25_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
    }

    if(part3){
      // conv_26
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_26_params_res4.batch_size, conv_26_params_res4.in_dim, conv_26_params_res4.in_channels,
          conv_26_params_res4.out_channels, conv_26_params_res4.out_dim,
          conv_26_params_res4.stride, 1, conv_26_params_res4.padding, conv_26_params_res4.kernel_size,
          conv_26_params_res4.out_stride,

          (elem_t*)conv_25_out_res4, (elem_t*)conv_26_w_res4, (acc_t*)conv_26_b_res4, (elem_t*)conv_26_out_res4,

          RELU, conv_26_params_res4.output_scale, 0,
          conv_26_params_res4.pool_size, 0, conv_26_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_27
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_27_params_res4.I, conv_27_params_res4.J, conv_27_params_res4.K, conv_27_params_res4.out_stride,
          (elem_t*)conv_26_out_res4, (elem_t*)conv_27_w_res4, (acc_t*)conv_27_b_res4, (elem_t*)conv_27_out_res4,
          NO_ACTIVATION, conv_27_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_24_out_res4
      // conv_28
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_28_params_res4.batch_size, conv_28_params_res4.in_dim, conv_28_params_res4.in_channels,
          conv_28_params_res4.out_channels, conv_28_params_res4.out_dim,
          conv_28_params_res4.stride, 1, conv_28_params_res4.padding, conv_28_params_res4.kernel_size,
          conv_28_params_res4.out_stride,

          (elem_t*)conv_24_out_res4, (elem_t*)conv_28_w_res4, (acc_t*)conv_28_b_res4, (elem_t*)conv_28_out_res4,

          NO_ACTIVATION, conv_28_params_res4.output_scale, 0,
          conv_28_params_res4.pool_size, 0, conv_28_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_27_params_res4.I, conv_27_params_res4.J,
          conv_27_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_28_out_res4,
          (elem_t*)conv_27_out_res4,
          (elem_t*)conv_27_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_29
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_29_params_res4.I, conv_29_params_res4.J, conv_29_params_res4.K, conv_29_params_res4.out_stride,
          (elem_t*)conv_27_out_res4, (elem_t*)conv_29_w_res4, (acc_t*)conv_29_b_res4, (elem_t*)conv_29_out_res4,
          RELU, conv_29_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_30
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_30_params_res4.batch_size, conv_30_params_res4.in_dim, conv_30_params_res4.in_channels,
          conv_30_params_res4.out_channels, conv_30_params_res4.out_dim,
          conv_30_params_res4.stride, 1, conv_30_params_res4.padding, conv_30_params_res4.kernel_size,
          conv_30_params_res4.out_stride,

          (elem_t*)conv_29_out_res4, (elem_t*)conv_30_w_res4, (acc_t*)conv_30_b_res4, (elem_t*)conv_30_out_res4,

          RELU, conv_30_params_res4.output_scale, 0,
          conv_30_params_res4.pool_size, 0, conv_30_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_31
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_31_params_res4.I, conv_31_params_res4.J, conv_31_params_res4.K, conv_31_params_res4.out_stride,
          (elem_t*)conv_30_out_res4, (elem_t*)conv_31_w_res4, (acc_t*)conv_31_b_res4, (elem_t*)conv_31_out_res4,
          NO_ACTIVATION, conv_31_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_31_params_res4.I, conv_31_params_res4.J,
          conv_31_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_27_out_res4,
          (elem_t*)conv_31_out_res4,
          (elem_t*)conv_31_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[8] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_32
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_32_params_res4.I, conv_32_params_res4.J, conv_32_params_res4.K, conv_32_params_res4.out_stride,
          (elem_t*)conv_31_out_res4, (elem_t*)conv_32_w_res4, (acc_t*)conv_32_b_res4, (elem_t*)conv_32_out_res4,
          RELU, conv_32_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_33
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_33_params_res4.batch_size, conv_33_params_res4.in_dim, conv_33_params_res4.in_channels,
          conv_33_params_res4.out_channels, conv_33_params_res4.out_dim,
          conv_33_params_res4.stride, 1, conv_33_params_res4.padding, conv_33_params_res4.kernel_size,
          conv_33_params_res4.out_stride,

          (elem_t*)conv_32_out_res4, (elem_t*)conv_33_w_res4, (acc_t*)conv_33_b_res4, (elem_t*)conv_33_out_res4,

          RELU, conv_33_params_res4.output_scale, 0,
          conv_33_params_res4.pool_size, 0, conv_33_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_34
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_34_params_res4.I, conv_34_params_res4.J, conv_34_params_res4.K, conv_34_params_res4.out_stride,
          (elem_t*)conv_33_out_res4, (elem_t*)conv_34_w_res4, (acc_t*)conv_34_b_res4, (elem_t*)conv_34_out_res4,
          NO_ACTIVATION, conv_34_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_34_params_res4.I, conv_34_params_res4.J,
          conv_34_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_31_out_res4,
          (elem_t*)conv_34_out_res4,
          (elem_t*)conv_34_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[9] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_35
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_35_params_res4.I, conv_35_params_res4.J, conv_35_params_res4.K, conv_35_params_res4.out_stride,
          (elem_t*)conv_34_out_res4, (elem_t*)conv_35_w_res4, (acc_t*)conv_35_b_res4, (elem_t*)conv_35_out_res4,
          RELU, conv_35_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_36
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_36_params_res4.batch_size, conv_36_params_res4.in_dim, conv_36_params_res4.in_channels,
          conv_36_params_res4.out_channels, conv_36_params_res4.out_dim,
          conv_36_params_res4.stride, 1, conv_36_params_res4.padding, conv_36_params_res4.kernel_size,
          conv_36_params_res4.out_stride,

          (elem_t*)conv_35_out_res4, (elem_t*)conv_36_w_res4, (acc_t*)conv_36_b_res4, (elem_t*)conv_36_out_res4,

          RELU, conv_36_params_res4.output_scale, 0,
          conv_36_params_res4.pool_size, 0, conv_36_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_37
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_37_params_res4.I, conv_37_params_res4.J, conv_37_params_res4.K, conv_37_params_res4.out_stride,
          (elem_t*)conv_36_out_res4, (elem_t*)conv_37_w_res4, (acc_t*)conv_37_b_res4, (elem_t*)conv_37_out_res4,
          NO_ACTIVATION, conv_37_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_37_params_res4.I, conv_37_params_res4.J,
          conv_37_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_34_out_res4,
          (elem_t*)conv_37_out_res4,
          (elem_t*)conv_37_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[10] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_38
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_38_params_res4.I, conv_38_params_res4.J, conv_38_params_res4.K, conv_38_params_res4.out_stride,
          (elem_t*)conv_37_out_res4, (elem_t*)conv_38_w_res4, (acc_t*)conv_38_b_res4, (elem_t*)conv_38_out_res4,
          RELU, conv_38_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_39
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_39_params_res4.batch_size, conv_39_params_res4.in_dim, conv_39_params_res4.in_channels,
          conv_39_params_res4.out_channels, conv_39_params_res4.out_dim,
          conv_39_params_res4.stride, 1, conv_39_params_res4.padding, conv_39_params_res4.kernel_size,
          conv_39_params_res4.out_stride,

          (elem_t*)conv_38_out_res4, (elem_t*)conv_39_w_res4, (acc_t*)conv_39_b_res4, (elem_t*)conv_39_out_res4,

          RELU, conv_39_params_res4.output_scale, 0,
          conv_39_params_res4.pool_size, 0, conv_39_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_40
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_40_params_res4.I, conv_40_params_res4.J, conv_40_params_res4.K, conv_40_params_res4.out_stride,
          (elem_t*)conv_39_out_res4, (elem_t*)conv_40_w_res4, (acc_t*)conv_40_b_res4, (elem_t*)conv_40_out_res4,
          NO_ACTIVATION, conv_40_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_40_params_res4.I, conv_40_params_res4.J,
          conv_40_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_37_out_res4,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_40_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[11] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_41
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_41_params_res4.I, conv_41_params_res4.J, conv_41_params_res4.K, conv_41_params_res4.out_stride,
          (elem_t*)conv_40_out_res4, (elem_t*)conv_41_w_res4, (acc_t*)conv_41_b_res4, (elem_t*)conv_41_out_res4,
          RELU, conv_41_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_42
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_42_params_res4.batch_size, conv_42_params_res4.in_dim, conv_42_params_res4.in_channels,
          conv_42_params_res4.out_channels, conv_42_params_res4.out_dim,
          conv_42_params_res4.stride, 1, conv_42_params_res4.padding, conv_42_params_res4.kernel_size,
          conv_42_params_res4.out_stride,

          (elem_t*)conv_41_out_res4, (elem_t*)conv_42_w_res4, (acc_t*)conv_42_b_res4, (elem_t*)conv_42_out_res4,

          RELU, conv_42_params_res4.output_scale, 0,
          conv_42_params_res4.pool_size, 0, conv_42_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_43
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_43_params_res4.I, conv_43_params_res4.J, conv_43_params_res4.K, conv_43_params_res4.out_stride,
          (elem_t*)conv_42_out_res4, (elem_t*)conv_43_w_res4, (acc_t*)conv_43_b_res4, (elem_t*)conv_43_out_res4,
          NO_ACTIVATION, conv_43_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_43_params_res4.I, conv_43_params_res4.J,
          conv_43_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_43_out_res4,
          (elem_t*)conv_43_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[12] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
    
        // conv_44
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_44_params_res4.I, conv_44_params_res4.J, conv_44_params_res4.K, conv_44_params_res4.out_stride,
            (elem_t*)conv_43_out_res4, (elem_t*)conv_44_w_res4, (acc_t*)conv_44_b_res4, (elem_t*)conv_44_out_res4,
            RELU, conv_44_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
    }

    if(part4){
      // conv_45
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_45_params_res4.batch_size, conv_45_params_res4.in_dim, conv_45_params_res4.in_channels,
          conv_45_params_res4.out_channels, conv_45_params_res4.out_dim,
          conv_45_params_res4.stride, 1, conv_45_params_res4.padding, conv_45_params_res4.kernel_size,
          conv_45_params_res4.out_stride,

          (elem_t*)conv_44_out_res4, (elem_t*)conv_45_w_res4, (acc_t*)conv_45_b_res4, (elem_t*)conv_45_out_res4,

          RELU, conv_45_params_res4.output_scale, 0,
          conv_45_params_res4.pool_size, 0, conv_45_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_46
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_46_params_res4.I, conv_46_params_res4.J, conv_46_params_res4.K, conv_46_params_res4.out_stride,
          (elem_t*)conv_45_out_res4, (elem_t*)conv_46_w_res4, (acc_t*)conv_46_b_res4, (elem_t*)conv_46_out_res4,
          NO_ACTIVATION, conv_46_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_43_out_res4
      // conv_47
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_47_params_res4.batch_size, conv_47_params_res4.in_dim, conv_47_params_res4.in_channels,
          conv_47_params_res4.out_channels, conv_47_params_res4.out_dim,
          conv_47_params_res4.stride, 1, conv_47_params_res4.padding, conv_47_params_res4.kernel_size,
          conv_47_params_res4.out_stride,

          (elem_t*)conv_43_out_res4, (elem_t*)conv_47_w_res4, (acc_t*)conv_47_b_res4, (elem_t*)conv_47_out_res4,

          NO_ACTIVATION, conv_47_params_res4.output_scale, 0,
          conv_47_params_res4.pool_size, 0, conv_47_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_46_params_res4.I, conv_46_params_res4.J,
          conv_46_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_47_out_res4,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_46_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[13] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_48
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_48_params_res4.I, conv_48_params_res4.J, conv_48_params_res4.K, conv_48_params_res4.out_stride,
          (elem_t*)conv_46_out_res4, (elem_t*)conv_48_w_res4, (acc_t*)conv_48_b_res4, (elem_t*)conv_48_out_res4,
          RELU, conv_48_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_49
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_49_params_res4.batch_size, conv_49_params_res4.in_dim, conv_49_params_res4.in_channels,
          conv_49_params_res4.out_channels, conv_49_params_res4.out_dim,
          conv_49_params_res4.stride, 1, conv_49_params_res4.padding, conv_49_params_res4.kernel_size,
          conv_49_params_res4.out_stride,

          (elem_t*)conv_48_out_res4, (elem_t*)conv_49_w_res4, (acc_t*)conv_49_b_res4, (elem_t*)conv_49_out_res4,

          RELU, conv_49_params_res4.output_scale, 0,
          conv_49_params_res4.pool_size, 0, conv_49_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_50
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_50_params_res4.I, conv_50_params_res4.J, conv_50_params_res4.K, conv_50_params_res4.out_stride,
          (elem_t*)conv_49_out_res4, (elem_t*)conv_50_w_res4, (acc_t*)conv_50_b_res4, (elem_t*)conv_50_out_res4,
          NO_ACTIVATION, conv_50_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_50_params_res4.I, conv_50_params_res4.J,
          conv_50_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_50_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[14] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_51
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_51_params_res4.I, conv_51_params_res4.J, conv_51_params_res4.K, conv_51_params_res4.out_stride,
          (elem_t*)conv_50_out_res4, (elem_t*)conv_51_w_res4, (acc_t*)conv_51_b_res4, (elem_t*)conv_51_out_res4,
          RELU, conv_51_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // conv_52
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_52_params_res4.batch_size, conv_52_params_res4.in_dim, conv_52_params_res4.in_channels,
          conv_52_params_res4.out_channels, conv_52_params_res4.out_dim,
          conv_52_params_res4.stride, 1, conv_52_params_res4.padding, conv_52_params_res4.kernel_size,
          conv_52_params_res4.out_stride,

          (elem_t*)conv_51_out_res4, (elem_t*)conv_52_w_res4, (acc_t*)conv_52_b_res4, (elem_t*)conv_52_out_res4,

          RELU, conv_52_params_res4.output_scale, 0,
          conv_52_params_res4.pool_size, 0, conv_52_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_53
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_53_params_res4.I, conv_53_params_res4.J, conv_53_params_res4.K, conv_53_params_res4.out_stride,
          (elem_t*)conv_52_out_res4, (elem_t*)conv_53_w_res4, (acc_t*)conv_53_b_res4, (elem_t*)conv_53_out_res4,
          NO_ACTIVATION, conv_53_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_53_params_res4.I, conv_53_params_res4.J,
          conv_53_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_53_out_res4,
          (elem_t*)conv_53_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[15] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
 
	gemmini_dram_util[group_id] = 0;         
      // Global averaging
      
      static elem_t average[4][2048] row_align(MAX_BLOCK_LEN);

      start = read_cycles();
      if(cid == 0)
          tiled_global_average_auto(conv_53_out_res4, average, conv_53_params_res4.batch_size,                         
              conv_53_params_res4.out_channels, conv_53_params_res4.out_dim, WS);
         

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif
	target_util = -1;
      // fc_54
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_54_params_res4.I, fc_54_params_res4.J, fc_54_params_res4.K, fc_54_params_res4.out_stride,
          (elem_t*)average, (elem_t*)fc_54_w_res4, (acc_t*)fc_54_b_res4, (elem_t*)fc_54_out_res4,
          NO_ACTIVATION, fc_54_params_res4.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
      //pthread_barrier_wait(barrier_res);
#endif  
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 54){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 70){
        cycles[cid][i] = resadd_cycles[i - 54];
      }
      else{
        if(i == 70) cycles[cid][i] = total_conv_cycles;
        if(i == 71) cycles[cid][i] = total_resadd_cycles;
        if(i == 72) cycles[cid][i] = total_conv_cycles + total_resadd_cycles + other_cycles;
      }
    }

    return cycles[cid];
#undef num_cycle
}

#ifndef BAREMETAL
uint64_t* resnet_batch_function_4(size_t cid, size_t group_id, bool part1, bool part2, bool part3, int target_util, pthread_barrier_t  *barrier_res){
#else
uint64_t* resnet_batch_function_4(size_t cid, size_t group_id, bool part1, bool part2, bool part3, int target_util){
#endif

#define num_cycle (20+34+16+3)
  static uint64_t cycles[NUM_CORE][num_cycle];
    uint64_t start, end;
    uint64_t total_conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, total_resadd_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[54];//[20];
    //uint64_t conv_cycles[34];
    uint64_t resadd_cycles[16];
   //uint64_t target_cycle = target_cycles;
//printf("Address of start for cid %d: %p\n", cid, &start);
//printf("Address of end for cid %d: %p\n", cid, &end);
//printf("barrier: %d\n", barrier_res);
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_res);
#endif

    // 1 batch processing
    if(part1){
      int batch_division = conv_1_params_res4.batch_size;
      int orow_divide = 2;
      int batch_divide = 1;
      for(int i = 0; i < batch_division; i++){
        // conv_1
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_1_params_res4.batch_size/batch_division, conv_1_params_res4.in_dim, conv_1_params_res4.in_channels,
            conv_1_params_res4.out_channels, conv_1_params_res4.out_dim,
            conv_1_params_res4.stride, 1, conv_1_params_res4.padding, conv_1_params_res4.kernel_size,
            conv_1_params_res4.out_stride,

            (elem_t*)image8_3, (elem_t*)conv_1_w_res4, (acc_t*)conv_1_b_res4, (elem_t*)conv_1_out_res4_pooled,

            RELU, conv_1_params_res4.output_scale, 0,
            conv_1_params_res4.pool_size, conv_1_params_res4.pool_stride, conv_1_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[0] = end - start;

#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif             
        // conv_2
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_2_params_res4.I/batch_division, conv_2_params_res4.J, conv_2_params_res4.K, conv_2_params_res4.out_stride,
            (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_2_w_res4, (acc_t*)conv_2_b_res4, (elem_t*)conv_2_out_res4,
            RELU, conv_2_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[1] = end - start;

      #if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_3
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_3_params_res4.batch_size/batch_division, conv_3_params_res4.in_dim, conv_3_params_res4.in_channels,
            conv_3_params_res4.out_channels, conv_3_params_res4.out_dim,
            conv_3_params_res4.stride, 1, conv_3_params_res4.padding, conv_3_params_res4.kernel_size,
            conv_3_params_res4.out_stride,

            (elem_t*)conv_2_out_res4, (elem_t*)conv_3_w_res4, (acc_t*)conv_3_b_res4, (elem_t*)conv_3_out_res4,

            RELU, conv_3_params_res4.output_scale, 0,
            conv_3_params_res4.pool_size, 0, conv_3_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[2] = end - start;

#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_4
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_4_params_res4.I/batch_division, conv_4_params_res4.J, conv_4_params_res4.K, conv_4_params_res4.out_stride,
            (elem_t*)conv_3_out_res4, (elem_t*)conv_4_w_res4, (acc_t*)conv_4_b_res4, (elem_t*)conv_4_out_res4,
            NO_ACTIVATION, conv_4_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // Downsampling conv_1_out_res4_pooled
        // conv_5
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_5_params_res4.I/batch_division, conv_5_params_res4.J, conv_5_params_res4.K, conv_5_params_res4.out_stride,
            (elem_t*)conv_1_out_res4_pooled, (elem_t*)conv_5_w_res4, (acc_t*)conv_5_b_res4, (elem_t*)conv_5_out_res4,
            NO_ACTIVATION, conv_5_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_4_params_res4.I/batch_division, conv_4_params_res4.J,
            conv_4_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_5_out_res4,
            (elem_t*)conv_4_out_res4,
            (elem_t*)conv_4_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[0] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_6
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_6_params_res4.I/batch_division, conv_6_params_res4.J, conv_6_params_res4.K, conv_6_params_res4.out_stride,
            (elem_t*)conv_4_out_res4, (elem_t*)conv_6_w_res4, (acc_t*)conv_6_b_res4, (elem_t*)conv_6_out_res4,
            RELU, conv_6_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_7
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_7_params_res4.batch_size/batch_division, conv_7_params_res4.in_dim, conv_7_params_res4.in_channels,
            conv_7_params_res4.out_channels, conv_7_params_res4.out_dim,
            conv_7_params_res4.stride, 1, conv_7_params_res4.padding, conv_7_params_res4.kernel_size,
            conv_7_params_res4.out_stride,

            (elem_t*)conv_6_out_res4, (elem_t*)conv_7_w_res4, (acc_t*)conv_7_b_res4, (elem_t*)conv_7_out_res4,

            RELU, conv_7_params_res4.output_scale, 0,
            conv_7_params_res4.pool_size, 0, conv_7_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_8
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_8_params_res4.I/batch_division, conv_8_params_res4.J, conv_8_params_res4.K, conv_8_params_res4.out_stride,
            (elem_t*)conv_7_out_res4, (elem_t*)conv_8_w_res4, (acc_t*)conv_8_b_res4, (elem_t*)conv_8_out_res4,
            NO_ACTIVATION, conv_8_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_8_params_res4.I/batch_division, conv_8_params_res4.J,
            conv_8_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_4_out_res4,
            (elem_t*)conv_8_out_res4,
            (elem_t*)conv_8_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[1] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_9
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_9_params_res4.I/batch_division, conv_9_params_res4.J, conv_9_params_res4.K, conv_9_params_res4.out_stride,
            (elem_t*)conv_8_out_res4, (elem_t*)conv_9_w_res4, (acc_t*)conv_9_b_res4, (elem_t*)conv_9_out_res4,
            RELU, conv_9_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_10
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_10_params_res4.batch_size/batch_division, conv_10_params_res4.in_dim, conv_10_params_res4.in_channels,
            conv_10_params_res4.out_channels, conv_10_params_res4.out_dim,
            conv_10_params_res4.stride, 1, conv_10_params_res4.padding, conv_10_params_res4.kernel_size,
            conv_10_params_res4.out_stride,

            (elem_t*)conv_9_out_res4, (elem_t*)conv_10_w_res4, (acc_t*)conv_10_b_res4, (elem_t*)conv_10_out_res4,

            RELU, conv_10_params_res4.output_scale, 0,
            conv_10_params_res4.pool_size, 0, conv_10_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_11
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_11_params_res4.I/batch_division, conv_11_params_res4.J, conv_11_params_res4.K, conv_11_params_res4.out_stride,
            (elem_t*)conv_10_out_res4, (elem_t*)conv_11_w_res4, (acc_t*)conv_11_b_res4, (elem_t*)conv_11_out_res4,
            NO_ACTIVATION, conv_11_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_11_params_res4.I/batch_division, conv_11_params_res4.J,
            conv_11_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_8_out_res4,
            (elem_t*)conv_11_out_res4,
            (elem_t*)conv_11_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[2] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_12
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_12_params_res4.I/batch_division, conv_12_params_res4.J, conv_12_params_res4.K, conv_12_params_res4.out_stride,
            (elem_t*)conv_11_out_res4, (elem_t*)conv_12_w_res4, (acc_t*)conv_12_b_res4, (elem_t*)conv_12_out_res4,
            RELU, conv_12_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
        // conv_13
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_13_params_res4.batch_size/batch_division, conv_13_params_res4.in_dim, conv_13_params_res4.in_channels,
            conv_13_params_res4.out_channels, conv_13_params_res4.out_dim,
            conv_13_params_res4.stride, 1, conv_13_params_res4.padding, conv_13_params_res4.kernel_size,
            conv_13_params_res4.out_stride,

            (elem_t*)conv_12_out_res4, (elem_t*)conv_13_w_res4, (acc_t*)conv_13_b_res4, (elem_t*)conv_13_out_res4,

            RELU, conv_13_params_res4.output_scale, 0,
            conv_13_params_res4.pool_size, 0, conv_13_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_14
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_14_params_res4.I/batch_division, conv_14_params_res4.J, conv_14_params_res4.K, conv_14_params_res4.out_stride,
            (elem_t*)conv_13_out_res4, (elem_t*)conv_14_w_res4, (acc_t*)conv_14_b_res4, (elem_t*)conv_14_out_res4,
            NO_ACTIVATION, conv_14_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // Downsampling conv_11_out_res4
        // conv_15
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_15_params_res4.batch_size/batch_division, conv_15_params_res4.in_dim, conv_15_params_res4.in_channels,
            conv_15_params_res4.out_channels, conv_15_params_res4.out_dim,
            conv_15_params_res4.stride, 1, conv_15_params_res4.padding, conv_15_params_res4.kernel_size,
            conv_15_params_res4.out_stride,

            (elem_t*)conv_11_out_res4, (elem_t*)conv_15_w_res4, (acc_t*)conv_15_b_res4, (elem_t*)conv_15_out_res4,

            NO_ACTIVATION, conv_15_params_res4.output_scale, 0,
            conv_15_params_res4.pool_size, 0, conv_15_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif        
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_14_params_res4.I/batch_division, conv_14_params_res4.J,
            conv_14_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_15_out_res4,
            (elem_t*)conv_14_out_res4,
            (elem_t*)conv_14_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[3] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_16
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_16_params_res4.I/batch_division, conv_16_params_res4.J, conv_16_params_res4.K, conv_16_params_res4.out_stride,
            (elem_t*)conv_14_out_res4, (elem_t*)conv_16_w_res4, (acc_t*)conv_16_b_res4, (elem_t*)conv_16_out_res4,
            RELU, conv_16_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
      } 
    }

    if(part2){   
      // 2 batch processing
      int batch_division = conv_1_params_res4.batch_size / 2;
      int orow_divide = 1;
      int batch_divide = 2;
      for(int i = 0; i < batch_division; i++){ 
        // conv_17
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_17_params_res4.batch_size/batch_division, conv_17_params_res4.in_dim, conv_17_params_res4.in_channels,
            conv_17_params_res4.out_channels, conv_17_params_res4.out_dim,
            conv_17_params_res4.stride, 1, conv_17_params_res4.padding, conv_17_params_res4.kernel_size,
            conv_17_params_res4.out_stride,

            (elem_t*)conv_16_out_res4, (elem_t*)conv_17_w_res4, (acc_t*)conv_17_b_res4, (elem_t*)conv_17_out_res4,

            RELU, conv_17_params_res4.output_scale, 0,
            conv_17_params_res4.pool_size, 0, conv_17_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_18
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_18_params_res4.I/batch_division, conv_18_params_res4.J, conv_18_params_res4.K, conv_18_params_res4.out_stride,
            (elem_t*)conv_17_out_res4, (elem_t*)conv_18_w_res4, (acc_t*)conv_18_b_res4, (elem_t*)conv_18_out_res4,
            NO_ACTIVATION, conv_18_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_18_params_res4.I/batch_division, conv_18_params_res4.J,
            conv_18_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_14_out_res4,
            (elem_t*)conv_18_out_res4,
            (elem_t*)conv_18_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[4] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_19
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_19_params_res4.I/batch_division, conv_19_params_res4.J, conv_19_params_res4.K, conv_19_params_res4.out_stride,
            (elem_t*)conv_18_out_res4, (elem_t*)conv_19_w_res4, (acc_t*)conv_19_b_res4, (elem_t*)conv_19_out_res4,
            RELU, conv_19_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // conv_20
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_20_params_res4.batch_size/batch_division, conv_20_params_res4.in_dim, conv_20_params_res4.in_channels,
            conv_20_params_res4.out_channels, conv_20_params_res4.out_dim,
            conv_20_params_res4.stride, 1, conv_20_params_res4.padding, conv_20_params_res4.kernel_size,
            conv_20_params_res4.out_stride,

            (elem_t*)conv_19_out_res4, (elem_t*)conv_20_w_res4, (acc_t*)conv_20_b_res4, (elem_t*)conv_20_out_res4,

            RELU, conv_20_params_res4.output_scale, 0,
            conv_20_params_res4.pool_size, 0, conv_20_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_21
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_21_params_res4.I/batch_division, conv_21_params_res4.J, conv_21_params_res4.K, conv_21_params_res4.out_stride,
            (elem_t*)conv_20_out_res4, (elem_t*)conv_21_w_res4, (acc_t*)conv_21_b_res4, (elem_t*)conv_21_out_res4,
            NO_ACTIVATION, conv_21_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_21_params_res4.I/batch_division, conv_21_params_res4.J,
            conv_21_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_18_out_res4,
            (elem_t*)conv_21_out_res4,
            (elem_t*)conv_21_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[5] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_22
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_22_params_res4.I/batch_division, conv_22_params_res4.J, conv_22_params_res4.K, conv_22_params_res4.out_stride,
            (elem_t*)conv_21_out_res4, (elem_t*)conv_22_w_res4, (acc_t*)conv_22_b_res4, (elem_t*)conv_22_out_res4,
            RELU, conv_22_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // conv_23
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_23_params_res4.batch_size/batch_division, conv_23_params_res4.in_dim, conv_23_params_res4.in_channels,
            conv_23_params_res4.out_channels, conv_23_params_res4.out_dim,
            conv_23_params_res4.stride, 1, conv_23_params_res4.padding, conv_23_params_res4.kernel_size,
            conv_23_params_res4.out_stride,

            (elem_t*)conv_22_out_res4, (elem_t*)conv_23_w_res4, (acc_t*)conv_23_b_res4, (elem_t*)conv_23_out_res4,

            RELU, conv_23_params_res4.output_scale, 0,
            conv_23_params_res4.pool_size, 0, conv_23_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_24
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_24_params_res4.I/batch_division, conv_24_params_res4.J, conv_24_params_res4.K, conv_24_params_res4.out_stride,
            (elem_t*)conv_23_out_res4, (elem_t*)conv_24_w_res4, (acc_t*)conv_24_b_res4, (elem_t*)conv_24_out_res4,
            NO_ACTIVATION, conv_24_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_24_params_res4.I/batch_division, conv_24_params_res4.J,
            conv_24_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_21_out_res4,
            (elem_t*)conv_24_out_res4,
            (elem_t*)conv_24_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[6] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_25
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_25_params_res4.I/batch_division, conv_25_params_res4.J, conv_25_params_res4.K, conv_25_params_res4.out_stride,
            (elem_t*)conv_24_out_res4, (elem_t*)conv_25_w_res4, (acc_t*)conv_25_b_res4, (elem_t*)conv_25_out_res4,
            RELU, conv_25_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
        // conv_26
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_26_params_res4.batch_size/batch_division, conv_26_params_res4.in_dim, conv_26_params_res4.in_channels,
            conv_26_params_res4.out_channels, conv_26_params_res4.out_dim,
            conv_26_params_res4.stride, 1, conv_26_params_res4.padding, conv_26_params_res4.kernel_size,
            conv_26_params_res4.out_stride,

            (elem_t*)conv_25_out_res4, (elem_t*)conv_26_w_res4, (acc_t*)conv_26_b_res4, (elem_t*)conv_26_out_res4,

            RELU, conv_26_params_res4.output_scale, 0,
            conv_26_params_res4.pool_size, 0, conv_26_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_27
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_27_params_res4.I/batch_division, conv_27_params_res4.J, conv_27_params_res4.K, conv_27_params_res4.out_stride,
            (elem_t*)conv_26_out_res4, (elem_t*)conv_27_w_res4, (acc_t*)conv_27_b_res4, (elem_t*)conv_27_out_res4,
            NO_ACTIVATION, conv_27_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Downsampling conv_24_out_res4
        // conv_28
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_28_params_res4.batch_size/batch_division, conv_28_params_res4.in_dim, conv_28_params_res4.in_channels,
            conv_28_params_res4.out_channels, conv_28_params_res4.out_dim,
            conv_28_params_res4.stride, 1, conv_28_params_res4.padding, conv_28_params_res4.kernel_size,
            conv_28_params_res4.out_stride,

            (elem_t*)conv_24_out_res4, (elem_t*)conv_28_w_res4, (acc_t*)conv_28_b_res4, (elem_t*)conv_28_out_res4,

            NO_ACTIVATION, conv_28_params_res4.output_scale, 0,
            conv_28_params_res4.pool_size, 0, conv_28_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_27_params_res4.I/batch_division, conv_27_params_res4.J,
            conv_27_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_28_out_res4,
            (elem_t*)conv_27_out_res4,
            (elem_t*)conv_27_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[7] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_29
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_29_params_res4.I/batch_division, conv_29_params_res4.J, conv_29_params_res4.K, conv_29_params_res4.out_stride,
            (elem_t*)conv_27_out_res4, (elem_t*)conv_29_w_res4, (acc_t*)conv_29_b_res4, (elem_t*)conv_29_out_res4,
            RELU, conv_29_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // conv_30
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_30_params_res4.batch_size/batch_division, conv_30_params_res4.in_dim, conv_30_params_res4.in_channels,
            conv_30_params_res4.out_channels, conv_30_params_res4.out_dim,
            conv_30_params_res4.stride, 1, conv_30_params_res4.padding, conv_30_params_res4.kernel_size,
            conv_30_params_res4.out_stride,

            (elem_t*)conv_29_out_res4, (elem_t*)conv_30_w_res4, (acc_t*)conv_30_b_res4, (elem_t*)conv_30_out_res4,

            RELU, conv_30_params_res4.output_scale, 0,
            conv_30_params_res4.pool_size, 0, conv_30_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_31
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_31_params_res4.I/batch_division, conv_31_params_res4.J, conv_31_params_res4.K, conv_31_params_res4.out_stride,
            (elem_t*)conv_30_out_res4, (elem_t*)conv_31_w_res4, (acc_t*)conv_31_b_res4, (elem_t*)conv_31_out_res4,
            NO_ACTIVATION, conv_31_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_31_params_res4.I/batch_division, conv_31_params_res4.J,
            conv_31_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_27_out_res4,
            (elem_t*)conv_31_out_res4,
            (elem_t*)conv_31_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[8] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_32
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_32_params_res4.I/batch_division, conv_32_params_res4.J, conv_32_params_res4.K, conv_32_params_res4.out_stride,
            (elem_t*)conv_31_out_res4, (elem_t*)conv_32_w_res4, (acc_t*)conv_32_b_res4, (elem_t*)conv_32_out_res4,
            RELU, conv_32_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // conv_33
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_33_params_res4.batch_size/batch_division, conv_33_params_res4.in_dim, conv_33_params_res4.in_channels,
            conv_33_params_res4.out_channels, conv_33_params_res4.out_dim,
            conv_33_params_res4.stride, 1, conv_33_params_res4.padding, conv_33_params_res4.kernel_size,
            conv_33_params_res4.out_stride,

            (elem_t*)conv_32_out_res4, (elem_t*)conv_33_w_res4, (acc_t*)conv_33_b_res4, (elem_t*)conv_33_out_res4,

            RELU, conv_33_params_res4.output_scale, 0,
            conv_33_params_res4.pool_size, 0, conv_33_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_34
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_34_params_res4.I/batch_division, conv_34_params_res4.J, conv_34_params_res4.K, conv_34_params_res4.out_stride,
            (elem_t*)conv_33_out_res4, (elem_t*)conv_34_w_res4, (acc_t*)conv_34_b_res4, (elem_t*)conv_34_out_res4,
            NO_ACTIVATION, conv_34_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_34_params_res4.I/batch_division, conv_34_params_res4.J,
            conv_34_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_31_out_res4,
            (elem_t*)conv_34_out_res4,
            (elem_t*)conv_34_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[9] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_35
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_35_params_res4.I/batch_division, conv_35_params_res4.J, conv_35_params_res4.K, conv_35_params_res4.out_stride,
            (elem_t*)conv_34_out_res4, (elem_t*)conv_35_w_res4, (acc_t*)conv_35_b_res4, (elem_t*)conv_35_out_res4,
            RELU, conv_35_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // conv_36
        start = read_cycles();
        tiled_conv_A_stride_auto_cid(
            conv_36_params_res4.batch_size/batch_division, conv_36_params_res4.in_dim, conv_36_params_res4.in_channels,
            conv_36_params_res4.out_channels, conv_36_params_res4.out_dim,
            conv_36_params_res4.stride, 1, conv_36_params_res4.padding, conv_36_params_res4.kernel_size,
            conv_36_params_res4.out_stride,

            (elem_t*)conv_35_out_res4, (elem_t*)conv_36_w_res4, (acc_t*)conv_36_b_res4, (elem_t*)conv_36_out_res4,

            RELU, conv_36_params_res4.output_scale, 0,
            conv_36_params_res4.pool_size, 0, conv_36_params_res4.pool_padding, false,

            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif        
            
        // conv_37
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_37_params_res4.I/batch_division, conv_37_params_res4.J, conv_37_params_res4.K, conv_37_params_res4.out_stride,
            (elem_t*)conv_36_out_res4, (elem_t*)conv_37_w_res4, (acc_t*)conv_37_b_res4, (elem_t*)conv_37_out_res4,
            NO_ACTIVATION, conv_37_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
        //pthread_barrier_wait(barrier_res);
#endif
            
        // Add residuals
        start = read_cycles();
        tiled_resadd_auto_cid(conv_37_params_res4.I/batch_division, conv_37_params_res4.J,
            conv_37_params_res4.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            (elem_t*)conv_34_out_res4,
            (elem_t*)conv_37_out_res4,
            (elem_t*)conv_37_out_res4,
            true,
            WS, orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_resadd_cycles += end - start;
        //resadd_cycles[10] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
            
        // conv_38
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_38_params_res4.I/batch_division, conv_38_params_res4.J, conv_38_params_res4.K, conv_38_params_res4.out_stride,
            (elem_t*)conv_37_out_res4, (elem_t*)conv_38_w_res4, (acc_t*)conv_38_b_res4, (elem_t*)conv_38_out_res4,
            RELU, conv_38_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        //conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
      } 
    }

    if(part3){  
      // 2 batch processing
      int batch_division = 1;
      int orow_divide = 1;
      int batch_divide = 2; 
      // conv_39
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_39_params_res4.batch_size/batch_division, conv_39_params_res4.in_dim, conv_39_params_res4.in_channels,
          conv_39_params_res4.out_channels, conv_39_params_res4.out_dim,
          conv_39_params_res4.stride, 1, conv_39_params_res4.padding, conv_39_params_res4.kernel_size,
          conv_39_params_res4.out_stride,

          (elem_t*)conv_38_out_res4, (elem_t*)conv_39_w_res4, (acc_t*)conv_39_b_res4, (elem_t*)conv_39_out_res4,

          RELU, conv_39_params_res4.output_scale, 0,
          conv_39_params_res4.pool_size, 0, conv_39_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_40
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_40_params_res4.I/batch_division, conv_40_params_res4.J, conv_40_params_res4.K, conv_40_params_res4.out_stride,
          (elem_t*)conv_39_out_res4, (elem_t*)conv_40_w_res4, (acc_t*)conv_40_b_res4, (elem_t*)conv_40_out_res4,
          NO_ACTIVATION, conv_40_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_40_params_res4.I/batch_division, conv_40_params_res4.J,
          conv_40_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_37_out_res4,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_40_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[11] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_41
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_41_params_res4.I/batch_division, conv_41_params_res4.J, conv_41_params_res4.K, conv_41_params_res4.out_stride,
          (elem_t*)conv_40_out_res4, (elem_t*)conv_41_w_res4, (acc_t*)conv_41_b_res4, (elem_t*)conv_41_out_res4,
          RELU, conv_41_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_42
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_42_params_res4.batch_size/batch_division, conv_42_params_res4.in_dim, conv_42_params_res4.in_channels,
          conv_42_params_res4.out_channels, conv_42_params_res4.out_dim,
          conv_42_params_res4.stride, 1, conv_42_params_res4.padding, conv_42_params_res4.kernel_size,
          conv_42_params_res4.out_stride,

          (elem_t*)conv_41_out_res4, (elem_t*)conv_42_w_res4, (acc_t*)conv_42_b_res4, (elem_t*)conv_42_out_res4,

          RELU, conv_42_params_res4.output_scale, 0,
          conv_42_params_res4.pool_size, 0, conv_42_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_43
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_43_params_res4.I/batch_division, conv_43_params_res4.J, conv_43_params_res4.K, conv_43_params_res4.out_stride,
          (elem_t*)conv_42_out_res4, (elem_t*)conv_43_w_res4, (acc_t*)conv_43_b_res4, (elem_t*)conv_43_out_res4,
          NO_ACTIVATION, conv_43_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_43_params_res4.I/batch_division, conv_43_params_res4.J,
          conv_43_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_40_out_res4,
          (elem_t*)conv_43_out_res4,
          (elem_t*)conv_43_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[12] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
    
        // conv_44
        start = read_cycles();
        tiled_matmul_nn_auto_cid(conv_44_params_res4.I/batch_division, conv_44_params_res4.J, conv_44_params_res4.K, conv_44_params_res4.out_stride,
            (elem_t*)conv_43_out_res4, (elem_t*)conv_44_w_res4, (acc_t*)conv_44_b_res4, (elem_t*)conv_44_out_res4,
            RELU, conv_44_params_res4.output_scale, 0, true,
            WS,
            orow_divide, batch_divide, cid, group_id, target_util);

        end = read_cycles();
        total_conv_cycles += end - start;
        conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
        pthread_barrier_wait(barrier_res);
#endif
      // conv_45
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_45_params_res4.batch_size/batch_division, conv_45_params_res4.in_dim, conv_45_params_res4.in_channels,
          conv_45_params_res4.out_channels, conv_45_params_res4.out_dim,
          conv_45_params_res4.stride, 1, conv_45_params_res4.padding, conv_45_params_res4.kernel_size,
          conv_45_params_res4.out_stride,

          (elem_t*)conv_44_out_res4, (elem_t*)conv_45_w_res4, (acc_t*)conv_45_b_res4, (elem_t*)conv_45_out_res4,

          RELU, conv_45_params_res4.output_scale, 0,
          conv_45_params_res4.pool_size, 0, conv_45_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_46
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_46_params_res4.I/batch_division, conv_46_params_res4.J, conv_46_params_res4.K, conv_46_params_res4.out_stride,
          (elem_t*)conv_45_out_res4, (elem_t*)conv_46_w_res4, (acc_t*)conv_46_b_res4, (elem_t*)conv_46_out_res4,
          NO_ACTIVATION, conv_46_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Downsampling conv_43_out_res4
      // conv_47
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_47_params_res4.batch_size/batch_division, conv_47_params_res4.in_dim, conv_47_params_res4.in_channels,
          conv_47_params_res4.out_channels, conv_47_params_res4.out_dim,
          conv_47_params_res4.stride, 1, conv_47_params_res4.padding, conv_47_params_res4.kernel_size,
          conv_47_params_res4.out_stride,

          (elem_t*)conv_43_out_res4, (elem_t*)conv_47_w_res4, (acc_t*)conv_47_b_res4, (elem_t*)conv_47_out_res4,

          NO_ACTIVATION, conv_47_params_res4.output_scale, 0,
          conv_47_params_res4.pool_size, 0, conv_47_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_46_params_res4.I/batch_division, conv_46_params_res4.J,
          conv_46_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_47_out_res4,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_46_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[13] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_48
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_48_params_res4.I/batch_division, conv_48_params_res4.J, conv_48_params_res4.K, conv_48_params_res4.out_stride,
          (elem_t*)conv_46_out_res4, (elem_t*)conv_48_w_res4, (acc_t*)conv_48_b_res4, (elem_t*)conv_48_out_res4,
          RELU, conv_48_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_49
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_49_params_res4.batch_size/batch_division, conv_49_params_res4.in_dim, conv_49_params_res4.in_channels,
          conv_49_params_res4.out_channels, conv_49_params_res4.out_dim,
          conv_49_params_res4.stride, 1, conv_49_params_res4.padding, conv_49_params_res4.kernel_size,
          conv_49_params_res4.out_stride,

          (elem_t*)conv_48_out_res4, (elem_t*)conv_49_w_res4, (acc_t*)conv_49_b_res4, (elem_t*)conv_49_out_res4,

          RELU, conv_49_params_res4.output_scale, 0,
          conv_49_params_res4.pool_size, 0, conv_49_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_50
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_50_params_res4.I/batch_division, conv_50_params_res4.J, conv_50_params_res4.K, conv_50_params_res4.out_stride,
          (elem_t*)conv_49_out_res4, (elem_t*)conv_50_w_res4, (acc_t*)conv_50_b_res4, (elem_t*)conv_50_out_res4,
          NO_ACTIVATION, conv_50_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_50_params_res4.I/batch_division, conv_50_params_res4.J,
          conv_50_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_46_out_res4,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_50_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[14] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_51
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_51_params_res4.I/batch_division, conv_51_params_res4.J, conv_51_params_res4.K, conv_51_params_res4.out_stride,
          (elem_t*)conv_50_out_res4, (elem_t*)conv_51_w_res4, (acc_t*)conv_51_b_res4, (elem_t*)conv_51_out_res4,
          RELU, conv_51_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // conv_52
      start = read_cycles();
      tiled_conv_A_stride_auto_cid(
          conv_52_params_res4.batch_size/batch_division, conv_52_params_res4.in_dim, conv_52_params_res4.in_channels,
          conv_52_params_res4.out_channels, conv_52_params_res4.out_dim,
          conv_52_params_res4.stride, 1, conv_52_params_res4.padding, conv_52_params_res4.kernel_size,
          conv_52_params_res4.out_stride,

          (elem_t*)conv_51_out_res4, (elem_t*)conv_52_w_res4, (acc_t*)conv_52_b_res4, (elem_t*)conv_52_out_res4,

          RELU, conv_52_params_res4.output_scale, 0,
          conv_52_params_res4.pool_size, 0, conv_52_params_res4.pool_padding, false,

          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif        
          
      // conv_53
      start = read_cycles();
      tiled_matmul_nn_auto_cid(conv_53_params_res4.I/batch_division, conv_53_params_res4.J, conv_53_params_res4.K, conv_53_params_res4.out_stride,
          (elem_t*)conv_52_out_res4, (elem_t*)conv_53_w_res4, (acc_t*)conv_53_b_res4, (elem_t*)conv_53_out_res4,
          NO_ACTIVATION, conv_53_params_res4.output_scale, 0, true,
          WS,
          orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
          
      // Add residuals
      start = read_cycles();
      tiled_resadd_auto_cid(conv_53_params_res4.I/batch_division, conv_53_params_res4.J,
          conv_53_params_res4.res_scale,
          MVIN_SCALE_IDENTITY,
          ACC_SCALE_IDENTITY,
          (elem_t*)conv_50_out_res4,
          (elem_t*)conv_53_out_res4,
          (elem_t*)conv_53_out_res4,
          true,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_resadd_cycles += end - start;
      resadd_cycles[15] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
      
      // Global averaging
      
      static elem_t average[4][2048] row_align(MAX_BLOCK_LEN);
/*
      start = read_cycles();
      if(cid == 1)
          tiled_global_average_auto(conv_53_out_res4, average, conv_53_params_res4.batch_size/batch_division,                         
              conv_53_params_res4.out_channels, conv_53_params_res4.out_dim, WS);
      else
	gemmini_dram_util[group_id] = 0;    

      end = read_cycles();
      other_cycles = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif
*/
	target_util = -1; // for last layer  
      // fc_54
      start = read_cycles();

      tiled_matmul_nn_auto_cid(fc_54_params_res4.I/batch_division, fc_54_params_res4.J, fc_54_params_res4.K, fc_54_params_res4.out_stride,
          (elem_t*)average, (elem_t*)fc_54_w_res4, (acc_t*)fc_54_b_res4, (elem_t*)fc_54_out_res4,
          NO_ACTIVATION, fc_54_params_res4.output_scale, 0, false,
          WS, orow_divide, batch_divide, cid, group_id, target_util);

      end = read_cycles();
      total_conv_cycles += end - start;
      conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
      pthread_barrier_wait(barrier_res);
#endif  
    }

    for(int i = 0; i < num_cycle; i++){
      if(i < 54){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 70){
        cycles[cid][i] = resadd_cycles[i - 54];
      }
      else{
        if(i == 70) cycles[cid][i] = total_conv_cycles;
        if(i == 71) cycles[cid][i] = total_resadd_cycles;
        if(i == 72) cycles[cid][i] = total_conv_cycles + total_resadd_cycles + other_cycles;
      }
    }

    return cycles[cid];
#undef num_cycle
}

