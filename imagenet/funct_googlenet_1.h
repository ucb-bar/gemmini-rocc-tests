
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "googlenet_params_1.h"
#include "images.h"

#ifndef BAREMETAL
#define THREAD_SYNC 1
#else
#define THREAD_SYNC 0
#endif

#ifndef BAREMETAL
uint64_t* googlenet_function_1(int cid, int orow_divide, int batch_divide, int target_util, pthread_barrier_t  *barrier_google){
#else
uint64_t* googlenet_function_1(int cid, int orow_divide, int batch_divide, int target_util){
#endif

#define num_cycle (58+11+3)

  static uint64_t cycles[num_proc][num_cycle];
 
    uint64_t start, end;
    uint64_t total_matmul_cycles = 0, total_conv_cycles = 0, total_pool_cycles = 0, conv_dw_cycles = 0, other_cycles = 0;
    uint64_t conv_cycles[58];
    uint64_t pool_cycles[11];

    //uint64_t target_cycle = target_cycles;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
  
    // conv_1
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_1_params_google1.batch_size, conv_1_params_google1.in_dim, conv_1_params_google1.in_channels,
        conv_1_params_google1.out_channels, conv_1_params_google1.out_dim,
        conv_1_params_google1.stride, 1, conv_1_params_google1.padding, conv_1_params_google1.kernel_size,
        conv_1_params_google1.out_stride,

        (elem_t*)images, (elem_t*)conv_1_w_google1, (acc_t*)conv_1_b_google1, (elem_t*)conv_1_out_google1_pooled,

        RELU, conv_1_params_google1.output_scale, 0,
        conv_1_params_google1.pool_size, conv_1_params_google1.pool_stride, conv_1_params_google1.pool_padding, true,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
        
    // conv_2
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_2_params_google1.I, conv_2_params_google1.J, conv_2_params_google1.K, conv_2_params_google1.out_stride,
        (elem_t*)conv_1_out_google1_pooled, (elem_t*)conv_2_w_google1, (acc_t*)conv_2_b_google1, (elem_t*)conv_2_out_google1,
        RELU, conv_2_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // conv_3
    start = read_cycles();
    tiled_conv_A_stride_auto_cid(
        conv_3_params_google1.batch_size, conv_3_params_google1.in_dim, conv_3_params_google1.in_channels,
        conv_3_params_google1.out_channels, conv_3_params_google1.out_dim,
        conv_3_params_google1.stride, 1, conv_3_params_google1.padding, conv_3_params_google1.kernel_size,
        conv_3_params_google1.out_stride,

        (elem_t*)conv_2_out_google1, (elem_t*)conv_3_w_google1, (acc_t*)conv_3_b_google1, (elem_t*)conv_3_out_google1,

        RELU, conv_3_params_google1.output_scale, 0,
        conv_3_params_google1.pool_size, 0, conv_3_params_google1.pool_padding, true,

        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif       
   // pool_9 for Inception 3a branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_9_params_google1.batch_size, pool_9_params_google1.out_channels, pool_9_params_google1.out_dim, pool_9_params_google1.out_dim_pooled,
		pool_9_params_google1.out_stride,
      pool_9_params_google1.pool_size, pool_9_params_google1.pool_stride, pool_9_params_google1.pool_padding,	
      (elem_t*) conv_3_out_google1_pooled, (elem_t*) pool_9_out_google1,
		orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
    //Inception 3a
	 //Branch 1
    // conv_4
    start = read_cycles();
   tiled_matmul_nn_auto_cid(conv_4_params_google1.I, conv_4_params_google1.J, conv_4_params_google1.K, 256 + 64,
        (elem_t*)conv_3_out_google1_pooled, (elem_t*)conv_4_w_google1, (acc_t*)conv_4_b_google1, (elem_t*) inception3a_out_google1,
        RELU, conv_4_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
	 // Branch 2
    // conv_5
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_5_params_google1.I, conv_5_params_google1.J, conv_5_params_google1.K, conv_5_params_google1.out_stride,
        (elem_t*)conv_3_out_google1_pooled, (elem_t*)conv_5_w_google1, (acc_t*)conv_5_b_google1, (elem_t*)conv_5_out_google1,
        RELU, conv_5_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_6
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_6_params_google1.batch_size, conv_6_params_google1.in_dim, conv_6_params_google1.in_channels,
			conv_6_params_google1.out_channels, conv_6_params_google1.out_dim,
			conv_6_params_google1.stride, 1, conv_6_params_google1.padding, conv_6_params_google1.kernel_size,
			256 + 64,

			(elem_t*)conv_5_out_google1, (elem_t*)conv_6_w_google1, (acc_t*)conv_6_b_google1, (elem_t*)((elem_t*)inception3a_out_google1 + 64), 

			RELU, conv_6_params_google1.output_scale, 0,
			conv_5_params_google1.pool_size, conv_5_params_google1.pool_stride, conv_5_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    // Branch 3
    // conv_7
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_7_params_google1.I, conv_7_params_google1.J, conv_7_params_google1.K,
			conv_7_params_google1.out_stride,
			(elem_t*) conv_3_out_google1_pooled, (elem_t*) conv_7_w_google1, (acc_t*) conv_7_b_google1, (elem_t*) conv_7_out_google1,
			RELU, conv_7_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_8
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_8_params_google1.batch_size, conv_8_params_google1.in_dim, conv_8_params_google1.in_channels,
			conv_8_params_google1.out_channels, conv_8_params_google1.out_dim,
			conv_8_params_google1.stride, 1, conv_8_params_google1.padding, conv_8_params_google1.kernel_size,
			256 + 64,

			(elem_t*)conv_7_out_google1, (elem_t*)conv_8_w_google1, (acc_t*)conv_8_b_google1, (elem_t*)((elem_t*)inception3a_out_google1 + 192), 

			RELU, conv_8_params_google1.output_scale, 0,
			conv_8_params_google1.pool_size, conv_8_params_google1.pool_stride, conv_8_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
       
    // Branch 4
    // conv_10
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_10_params_google1.I, conv_10_params_google1.J, conv_10_params_google1.K, 256 + 64,
        (elem_t*)pool_9_out_google1, (elem_t*)conv_10_w_google1, (acc_t*)conv_10_b_google1, (elem_t*)inception3a_out_google1 + 224,
        RELU, conv_10_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
// pool_16 for Inception 3a branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_16_params_google1.batch_size, pool_16_params_google1.out_channels, pool_16_params_google1.out_dim, pool_16_params_google1.out_dim_pooled,
		pool_16_params_google1.out_stride,
      pool_16_params_google1.pool_size, pool_16_params_google1.pool_stride, pool_16_params_google1.pool_padding,	
      (elem_t*) inception3a_out_google1, (elem_t*) pool_16_out_google1,
		orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
	 //Inception 3b
	 //Branch 1
    // conv_11
    start = read_cycles();

    tiled_matmul_nn_auto_cid(conv_11_params_google1.I, conv_11_params_google1.J, conv_11_params_google1.K, 480,
        (elem_t*) inception3a_out_google1, (elem_t*)conv_11_w_google1, (acc_t*)conv_11_b_google1, (elem_t*) pool_18_in_google1,
        NO_ACTIVATION, conv_11_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_12
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_12_params_google1.I, conv_12_params_google1.J, conv_12_params_google1.K, conv_12_params_google1.out_stride,
        (elem_t*) inception3a_out_google1, (elem_t*)conv_12_w_google1, (acc_t*)conv_12_b_google1, (elem_t*)conv_12_out_google1,
        NO_ACTIVATION, conv_12_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_13
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_13_params_google1.batch_size, conv_13_params_google1.in_dim, conv_13_params_google1.in_channels,
			conv_13_params_google1.out_channels, conv_13_params_google1.out_dim,
			conv_13_params_google1.stride, 1, conv_13_params_google1.padding, conv_13_params_google1.kernel_size,
			480,

			(elem_t*)conv_12_out_google1, (elem_t*)conv_13_w_google1, (acc_t*)conv_13_b_google1, (elem_t*)((elem_t*)(pool_18_in_google1) + 128),

			RELU, conv_13_params_google1.output_scale, 0,
			1, 1, 0, false,
      //pool_18_params_google1.pool_size, pool_18_params_google1.pool_stride, pool_18_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[11] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_14
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_14_params_google1.I, conv_14_params_google1.J, conv_14_params_google1.K,
			conv_14_params_google1.out_stride,
			(elem_t*) inception3a_out_google1, (elem_t*) conv_14_w_google1, (acc_t*) conv_14_b_google1, (elem_t*) conv_14_out_google1,
			RELU, conv_14_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[12] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_15
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_15_params_google1.batch_size, conv_15_params_google1.in_dim, conv_15_params_google1.in_channels,
			conv_15_params_google1.out_channels, conv_15_params_google1.out_dim,
			conv_15_params_google1.stride, 1, conv_15_params_google1.padding, conv_15_params_google1.kernel_size,
			480,

			(elem_t*)conv_14_out_google1, (elem_t*)conv_15_w_google1, (acc_t*)conv_15_b_google1, (elem_t*)((elem_t*) pool_18_in_google1 + 320),

			RELU, conv_15_params_google1.output_scale, 0,
			//pool_18_params_google1.pool_size, pool_18_params_google1.pool_stride, pool_18_params_google1.pool_padding, true,
      1, 1, 0, false,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[13] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // Branch 4
    // conv_17
    start = read_cycles();

    tiled_matmul_nn_auto_cid(conv_17_params_google1.I, conv_17_params_google1.J, conv_17_params_google1.K, 480,
        (elem_t*) pool_16_out_google1, (elem_t*)conv_17_w_google1, (acc_t*)conv_17_b_google1, (elem_t*) ((elem_t*) pool_18_in_google1 + 416),
        NO_ACTIVATION, conv_17_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[14] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        


// pool_18
    start = read_cycles();
    tiled_pool_auto_cid(pool_18_params_google1.batch_size, pool_18_params_google1.out_channels, pool_18_params_google1.out_dim, pool_18_params_google1.out_dim_pooled,
		   pool_18_params_google1.out_stride,
        pool_18_params_google1.pool_size, pool_18_params_google1.pool_stride, pool_18_params_google1.pool_padding,
			(elem_t*) pool_18_in_google1, (elem_t*) pool_18_out_google1,
			orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

	 //pool for inception 4a branch 4 input
    start = read_cycles();
    tiled_pool_auto_cid(pool_24_params_google1.batch_size, pool_24_params_google1.out_channels, pool_24_params_google1.out_dim, pool_24_params_google1.out_dim_pooled,
		   pool_24_params_google1.out_stride,
        pool_24_params_google1.pool_size, pool_24_params_google1.pool_stride, pool_24_params_google1.pool_padding,
			(elem_t*) pool_18_out_google1, (elem_t*) pool_24_out_google1,
			orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

	 //Inception 4a
	 //Branch 1
    // conv_19
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_19_params_google1.I, conv_19_params_google1.J, conv_19_params_google1.K, 512 + 64,
        (elem_t*)pool_18_out_google1, (elem_t*)conv_19_w_google1, (acc_t*)conv_19_b_google1, (elem_t*)inception4a_out_google1,
        RELU, conv_19_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[15] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_20
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_20_params_google1.I, conv_20_params_google1.J, conv_20_params_google1.K, conv_20_params_google1.out_stride,
        (elem_t*)pool_18_out_google1, (elem_t*)conv_20_w_google1, (acc_t*)conv_20_b_google1, (elem_t*)conv_20_out_google1,
        RELU, conv_20_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[16] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_21
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_21_params_google1.batch_size, conv_21_params_google1.in_dim, conv_21_params_google1.in_channels,
			conv_21_params_google1.out_channels, conv_21_params_google1.out_dim,
			conv_21_params_google1.stride, 1, conv_21_params_google1.padding, conv_21_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_20_out_google1, (elem_t*)conv_21_w_google1, (acc_t*)conv_21_b_google1, (elem_t*)((elem_t*)inception4a_out_google1 + 192),

			RELU, conv_21_params_google1.output_scale, 0,
			conv_21_params_google1.pool_size, conv_21_params_google1.pool_stride, conv_21_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[17] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_22
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_22_params_google1.I, conv_22_params_google1.J, conv_22_params_google1.K,
			conv_22_params_google1.out_stride,
			(elem_t*) pool_18_out_google1, (elem_t*) conv_22_w_google1, (acc_t*) conv_22_b_google1, (elem_t*) conv_22_out_google1,
			RELU, conv_22_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[18] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_23
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_23_params_google1.batch_size, conv_23_params_google1.in_dim, conv_23_params_google1.in_channels,
			conv_23_params_google1.out_channels, conv_23_params_google1.out_dim,
			conv_23_params_google1.stride, 1, conv_23_params_google1.padding, conv_23_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_22_out_google1, (elem_t*)conv_23_w_google1, (acc_t*)conv_23_b_google1, (elem_t*)((elem_t*)inception4a_out_google1 + 400),

			RELU, conv_23_params_google1.output_scale, 0,
			conv_23_params_google1.pool_size, conv_23_params_google1.pool_stride, conv_23_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[19] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // Branch 4
    // conv_25
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_25_params_google1.I, conv_25_params_google1.J, conv_25_params_google1.K, 512 + 64,
        (elem_t*)pool_24_out_google1, (elem_t*)conv_25_w_google1, (acc_t*)conv_25_b_google1, (elem_t*)inception4a_out_google1 + 448,
        RELU, conv_25_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[20] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

    
    // pool_31 for Inception 3a branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_31_params_google1.batch_size, pool_31_params_google1.out_channels, pool_31_params_google1.out_dim, pool_31_params_google1.out_dim_pooled,
		pool_31_params_google1.out_stride,
      pool_31_params_google1.pool_size, pool_31_params_google1.pool_stride, pool_31_params_google1.pool_padding,	
      (elem_t*) inception4a_out_google1, (elem_t*) pool_31_out_google1,
		orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
	 //Inception 4b
	 // Branch 1
    // conv_26
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_26_params_google1.I, conv_26_params_google1.J, conv_26_params_google1.K, 512 + 64,
        (elem_t*)inception4a_out_google1, (elem_t*)conv_26_w_google1, (acc_t*)conv_26_b_google1, (elem_t*)inception4b_out_google1,
        RELU, conv_26_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[21] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_27
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_27_params_google1.I, conv_27_params_google1.J, conv_27_params_google1.K, conv_27_params_google1.out_stride,
        (elem_t*)inception4a_out_google1, (elem_t*)conv_27_w_google1, (acc_t*)conv_27_b_google1, (elem_t*)conv_27_out_google1,
        NO_ACTIVATION, conv_27_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[22] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif          
    // conv_28
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_28_params_google1.batch_size, conv_28_params_google1.in_dim, conv_28_params_google1.in_channels,
			conv_28_params_google1.out_channels, conv_28_params_google1.out_dim,
			conv_28_params_google1.stride, 1, conv_28_params_google1.padding, conv_28_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_27_out_google1, (elem_t*)conv_28_w_google1, (acc_t*)conv_28_b_google1, (elem_t*)((elem_t*)inception4b_out_google1 + 160),

			RELU, conv_28_params_google1.output_scale, 0,
			conv_27_params_google1.pool_size, conv_27_params_google1.pool_stride, conv_27_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[23] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_29
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_29_params_google1.I, conv_29_params_google1.J, conv_29_params_google1.K,
			conv_29_params_google1.out_stride,
			(elem_t*) inception4a_out_google1, (elem_t*) conv_29_w_google1, (acc_t*) conv_29_b_google1, (elem_t*) conv_29_out_google1,
			RELU, conv_29_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[24] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_30
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_30_params_google1.batch_size, conv_30_params_google1.in_dim, conv_30_params_google1.in_channels,
			conv_30_params_google1.out_channels, conv_30_params_google1.out_dim,
			conv_30_params_google1.stride, 1, conv_30_params_google1.padding, conv_30_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_29_out_google1, (elem_t*)conv_30_w_google1, (acc_t*)conv_30_b_google1, (elem_t*)((elem_t*)inception4b_out_google1 + 384),

			RELU, conv_30_params_google1.output_scale, 0,
			conv_30_params_google1.pool_size, conv_30_params_google1.pool_stride, conv_30_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
//pthread_barrier_wait(barrier_google);
    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[25] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // Branch 4
    // conv_32
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_32_params_google1.I, conv_32_params_google1.J, conv_32_params_google1.K, 512 + 64,
        (elem_t*)pool_31_out_google1, (elem_t*)conv_32_w_google1, (acc_t*)conv_32_b_google1, (elem_t*)inception4b_out_google1 + 448,
        RELU, conv_32_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[26] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

// pool_38 for Inception 4b branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_38_params_google1.batch_size, pool_38_params_google1.out_channels, pool_38_params_google1.out_dim, pool_38_params_google1.out_dim_pooled,
		pool_38_params_google1.out_stride,
      pool_38_params_google1.pool_size, pool_38_params_google1.pool_stride, pool_38_params_google1.pool_padding,	
      (elem_t*) inception4b_out_google1, (elem_t*) pool_38_out_google1,
		orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[5] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

	 //Inception 4c
	 // Branch 1
    // conv_33
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_33_params_google1.I, conv_33_params_google1.J, conv_33_params_google1.K, 512 + 64,
        (elem_t*)inception4b_out_google1, (elem_t*)conv_33_w_google1, (acc_t*)conv_33_b_google1, (elem_t*)inception4c_out_google1,
        RELU, conv_33_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[27] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_34
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_34_params_google1.I, conv_34_params_google1.J, conv_34_params_google1.K, conv_34_params_google1.out_stride,
        (elem_t*)inception4b_out_google1, (elem_t*)conv_34_w_google1, (acc_t*)conv_34_b_google1, (elem_t*)conv_34_out_google1,
        NO_ACTIVATION, conv_34_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[28] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_35
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_35_params_google1.batch_size, conv_35_params_google1.in_dim, conv_35_params_google1.in_channels,
			conv_35_params_google1.out_channels, conv_35_params_google1.out_dim,
			conv_35_params_google1.stride, 1, conv_35_params_google1.padding, conv_35_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_34_out_google1, (elem_t*)conv_35_w_google1, (acc_t*)conv_35_b_google1, (elem_t*)((elem_t*)inception4c_out_google1 + 128), 

			RELU, conv_35_params_google1.output_scale, 0,
			conv_35_params_google1.pool_size, conv_35_params_google1.pool_stride, conv_35_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[29] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_36
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_36_params_google1.I, conv_36_params_google1.J, conv_36_params_google1.K,
			conv_36_params_google1.out_stride,
			(elem_t*) inception4b_out_google1, (elem_t*) conv_36_w_google1, (acc_t*) conv_36_b_google1, (elem_t*) conv_36_out_google1,
			RELU, conv_36_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[30] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_37
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_37_params_google1.batch_size, conv_37_params_google1.in_dim, conv_37_params_google1.in_channels,
			conv_37_params_google1.out_channels, conv_37_params_google1.out_dim,
			conv_37_params_google1.stride, 1, conv_37_params_google1.padding, conv_37_params_google1.kernel_size,
			512 + 64,

			(elem_t*)conv_36_out_google1, (elem_t*)conv_37_w_google1, (acc_t*)conv_37_b_google1, (elem_t*)((elem_t*)inception4c_out_google1 + 384),

			RELU, conv_37_params_google1.output_scale, 0,
			conv_37_params_google1.pool_size, conv_37_params_google1.pool_stride, conv_37_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
//pthread_barrier_wait(barrier_google);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[31] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // Branch 4
    // conv_39
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_39_params_google1.I, conv_39_params_google1.J, conv_39_params_google1.K, 512 + 64,
        (elem_t*)pool_38_out_google1, (elem_t*)conv_39_w_google1, (acc_t*)conv_39_b_google1, (elem_t*)inception4c_out_google1 + 448,
        NO_ACTIVATION, conv_39_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[32] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif       

// pool_45 for Inception 4c branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_45_params_google1.batch_size, pool_45_params_google1.out_channels, pool_45_params_google1.out_dim, pool_45_params_google1.out_dim_pooled,
		pool_45_params_google1.out_stride,
      pool_45_params_google1.pool_size, pool_45_params_google1.pool_stride, pool_45_params_google1.pool_padding,	
      (elem_t*) inception4c_out_google1, (elem_t*) pool_45_out_google1,
		orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[6] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

    //Inception 4d
	 // Branch 1
    // conv_40
     start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_40_params_google1.I, conv_40_params_google1.J, conv_40_params_google1.K, 528,
        (elem_t*)inception4c_out_google1, (elem_t*)conv_40_w_google1, (acc_t*)conv_40_b_google1, (elem_t*) inception4d_out_google1,
        NO_ACTIVATION, conv_40_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[33] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_41
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_41_params_google1.I, conv_41_params_google1.J, conv_41_params_google1.K, conv_41_params_google1.out_stride,
        (elem_t*)inception4c_out_google1, (elem_t*)conv_41_w_google1, (acc_t*)conv_41_b_google1, (elem_t*)conv_41_out_google1,
        RELU, conv_41_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[34] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_42
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_42_params_google1.batch_size, conv_42_params_google1.in_dim, conv_42_params_google1.in_channels,
			conv_42_params_google1.out_channels, conv_42_params_google1.out_dim,
			conv_42_params_google1.stride, 1, conv_42_params_google1.padding, conv_42_params_google1.kernel_size,
			528,

			(elem_t*)conv_41_out_google1, (elem_t*)conv_42_w_google1, (acc_t*)conv_42_b_google1, (elem_t*)((elem_t*)inception4d_out_google1 + 112),

			RELU, conv_42_params_google1.output_scale, 0,
			conv_42_params_google1.pool_size, conv_42_params_google1.pool_stride, conv_42_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[35] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_43
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_43_params_google1.I, conv_43_params_google1.J, conv_43_params_google1.K,
			conv_43_params_google1.out_stride,
			(elem_t*) inception4c_out_google1, (elem_t*) conv_43_w_google1, (acc_t*) conv_43_b_google1, (elem_t*) conv_43_out_google1,
			RELU, conv_43_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[36] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_44
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_44_params_google1.batch_size, conv_44_params_google1.in_dim, conv_44_params_google1.in_channels,
			conv_44_params_google1.out_channels, conv_44_params_google1.out_dim,
			conv_44_params_google1.stride, 1, conv_44_params_google1.padding, conv_44_params_google1.kernel_size,
			528,

			(elem_t*)conv_43_out_google1, (elem_t*)conv_44_w_google1, (acc_t*)conv_44_b_google1, (elem_t*)((elem_t*)inception4d_out_google1 + 400),

			RELU, conv_44_params_google1.output_scale, 0,
			conv_44_params_google1.pool_size, conv_44_params_google1.pool_stride, conv_44_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
   // pthread_barrier_wait(barrier_google);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[37] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
        
    // Branch 4
    // conv_46
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_46_params_google1.I, conv_46_params_google1.J, conv_46_params_google1.K,
			528,
			(elem_t*) pool_45_out_google1, (elem_t*) conv_46_w_google1, (acc_t*) conv_46_b_google1, (elem_t*) inception4d_out_google1 + 464,
			RELU, conv_46_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[38] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
// pool_52 for Inception 4d branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_52_params_google1.batch_size, pool_52_params_google1.out_channels, pool_52_params_google1.out_dim, pool_52_params_google1.out_dim_pooled,
		pool_52_params_google1.out_stride,
      pool_52_params_google1.pool_size, pool_52_params_google1.pool_stride, pool_52_params_google1.pool_padding,	
      (elem_t*) inception4d_out_google1, (elem_t*) pool_52_out_google1,
		orow_divide, batch_divide, cid, target_util);
gemmini_fence();
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[7] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

	 //Inception 4e
	 //Branch 1
    // conv_47
    start = read_cycles();
/*
    tiled_conv_A_stride_auto_cid(
			conv_47_params_google1.batch_size, conv_47_params_google1.in_dim, conv_47_params_google1.in_channels,
			conv_47_params_google1.out_channels, conv_47_params_google1.out_dim,
			conv_47_params_google1.stride, 1, conv_47_params_google1.padding, conv_47_params_google1.kernel_size,
			832,

			(elem_t*) inception4d_out_google1, (elem_t*)conv_47_w_google1, (acc_t*)conv_47_b_google1, (elem_t*) pool_54_out_google1,

			RELU, conv_47_params_google1.output_scale, 0,
			pool_54_params_google1.pool_size, pool_54_params_google1.pool_stride, pool_54_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
*/
    tiled_matmul_nn_auto_cid(conv_47_params_google1.I, conv_47_params_google1.J, conv_47_params_google1.K, 832,
        (elem_t*)inception4d_out_google1, (elem_t*)conv_47_w_google1, (acc_t*)conv_47_b_google1, (elem_t*) pool_54_in_google1,
        NO_ACTIVATION, conv_47_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[39] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

	 // Branch 2
    // conv_48
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_48_params_google1.I, conv_48_params_google1.J, conv_48_params_google1.K, conv_48_params_google1.out_stride,
        (elem_t*) inception4d_out_google1, (elem_t*)conv_48_w_google1, (acc_t*)conv_48_b_google1, (elem_t*)conv_48_out_google1,
        NO_ACTIVATION, conv_48_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[40] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif  
    // conv_13
    start = read_cycles();

    tiled_conv_A_stride_auto_cid(
			conv_49_params_google1.batch_size, conv_49_params_google1.in_dim, conv_49_params_google1.in_channels,
			conv_49_params_google1.out_channels, conv_49_params_google1.out_dim,
			conv_49_params_google1.stride, 1, conv_49_params_google1.padding, conv_49_params_google1.kernel_size,
			832,

			(elem_t*)conv_48_out_google1, (elem_t*)conv_49_w_google1, (acc_t*)conv_49_b_google1, (elem_t*)((elem_t*)(pool_54_in_google1) + 256),

			RELU, conv_49_params_google1.output_scale, 0,
			1, 1, 0, false,
      //pool_54_params_google1.pool_size, pool_54_params_google1.pool_stride, pool_54_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[41] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
	 // Branch 3
    // conv_50
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_50_params_google1.I, conv_50_params_google1.J, conv_50_params_google1.K,
			conv_50_params_google1.out_stride,
			(elem_t*) inception4d_out_google1, (elem_t*) conv_50_w_google1, (acc_t*) conv_50_b_google1, (elem_t*) conv_50_out_google1,
			RELU, conv_50_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[42] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_51
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_51_params_google1.batch_size, conv_51_params_google1.in_dim, conv_51_params_google1.in_channels,
			conv_51_params_google1.out_channels, conv_51_params_google1.out_dim,
			conv_51_params_google1.stride, 1, conv_51_params_google1.padding, conv_51_params_google1.kernel_size,
			832,

			(elem_t*)conv_50_out_google1, (elem_t*)conv_51_w_google1, (acc_t*)conv_51_b_google1, (elem_t*)((elem_t*) pool_54_in_google1 + 576),

			RELU, conv_51_params_google1.output_scale, 0,
			1, 1, 0, false,
      //pool_54_params_google1.pool_size, pool_54_params_google1.pool_stride, pool_54_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
 //pthread_barrier_wait(barrier_google);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[43] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    // Branch 4
    // conv_53
    start = read_cycles();
/*
    tiled_conv_A_stride_auto_cid(
        conv_53_params_google1.batch_size, conv_53_params_google1.in_dim, conv_53_params_google1.in_channels,
        conv_53_params_google1.out_channels, conv_53_params_google1.out_dim,
        conv_53_params_google1.stride, 1, conv_53_params_google1.padding, conv_53_params_google1.kernel_size,
        832,

        (elem_t*)pool_52_out_google1, (elem_t*)conv_53_w_google1, (acc_t*)conv_53_b_google1, (elem_t*)((elem_t*) pool_54_out_google1 + 704),

        RELU, conv_53_params_google1.output_scale, 0,
  		  pool_54_params_google1.pool_size, pool_54_params_google1.pool_stride, pool_54_params_google1.pool_padding, true,

        WS, orow_divide, batch_divide, cid, target_util);
*/
    tiled_matmul_nn_auto_cid(conv_53_params_google1.I, conv_53_params_google1.J, conv_53_params_google1.K, 832,
        (elem_t*)inception4d_out_google1, (elem_t*)conv_53_w_google1, (acc_t*)conv_53_b_google1, (elem_t*)((elem_t*) (pool_54_in_google1) + 704),
        NO_ACTIVATION, conv_53_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[44] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif       

// pool
    start = read_cycles();
    tiled_pool_auto_cid(pool_54_params_google1.batch_size, pool_54_params_google1.out_channels, pool_54_params_google1.out_dim, pool_54_params_google1.out_dim_pooled,
		   pool_54_params_google1.out_stride,
        pool_54_params_google1.pool_size, pool_54_params_google1.pool_stride, pool_54_params_google1.pool_padding,
			(elem_t*) pool_54_out_google1, (elem_t*) pool_54_out_google1,
			orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[8] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif   


  //pool for inception 5a branch 4 input
    start = read_cycles();
    tiled_pool_auto_cid(pool_60_params_google1.batch_size, pool_60_params_google1.out_channels, pool_60_params_google1.out_dim, pool_60_params_google1.out_dim_pooled,
		   pool_60_params_google1.out_stride,
        pool_60_params_google1.pool_size, pool_60_params_google1.pool_stride, pool_60_params_google1.pool_padding,
			(elem_t*) pool_54_out_google1, (elem_t*) pool_60_out_google1,
			orow_divide, batch_divide, cid, target_util);
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[9] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif       

    // Branch 4
    // conv_61
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_61_params_google1.I, conv_61_params_google1.J, conv_61_params_google1.K, 832,
        (elem_t*) pool_60_out_google1, (elem_t*)conv_61_w_google1, (acc_t*)conv_61_b_google1, (elem_t*) inception5a_out_google1 + 704,
        NO_ACTIVATION, conv_61_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[45] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        

	 // Inception 5a
	 // Branch 1
    // conv_55
    start = read_cycles();
	  tiled_matmul_nn_auto_cid(conv_55_params_google1.I, conv_55_params_google1.J, conv_55_params_google1.K,
			832,
			(elem_t*) pool_54_out_google1, (elem_t*) conv_55_w_google1, (acc_t*) conv_55_b_google1, (elem_t*) inception5a_out_google1,
			RELU, conv_55_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[46] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
	 // Branch 2
    // conv_56
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_56_params_google1.I, conv_56_params_google1.J, conv_56_params_google1.K, conv_56_params_google1.out_stride,
        (elem_t*)pool_54_out_google1, (elem_t*)conv_56_w_google1, (acc_t*)conv_56_b_google1, (elem_t*)conv_56_out_google1,
        NO_ACTIVATION, conv_56_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[47] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif

    // conv_57
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_57_params_google1.batch_size, conv_57_params_google1.in_dim, conv_57_params_google1.in_channels,
			conv_57_params_google1.out_channels, conv_57_params_google1.out_dim,
			conv_57_params_google1.stride, 1, conv_57_params_google1.padding, conv_57_params_google1.kernel_size,
			832,

			(elem_t*)conv_56_out_google1, (elem_t*)conv_57_w_google1, (acc_t*)conv_57_b_google1, (elem_t*)((elem_t*)inception5a_out_google1 + 256),

			RELU, conv_57_params_google1.output_scale, 0,
			conv_57_params_google1.pool_size, conv_57_params_google1.pool_stride, conv_57_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[48] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
	 // Branch 3
    // conv_58
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_58_params_google1.I, conv_58_params_google1.J, conv_58_params_google1.K,
			conv_58_params_google1.out_stride,
			(elem_t*) pool_54_out_google1, (elem_t*) conv_58_w_google1, (acc_t*) conv_58_b_google1, (elem_t*) conv_58_out_google1,
			RELU, conv_58_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[49] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // conv_59
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_59_params_google1.batch_size, conv_59_params_google1.in_dim, conv_59_params_google1.in_channels,
			conv_59_params_google1.out_channels, conv_59_params_google1.out_dim,
			conv_59_params_google1.stride, 1, conv_59_params_google1.padding, conv_59_params_google1.kernel_size,
			832,

			(elem_t*)conv_58_out_google1, (elem_t*)conv_59_w_google1, (acc_t*)conv_59_b_google1, (elem_t*)((elem_t*)inception5a_out_google1 + 576),

			RELU, conv_59_params_google1.output_scale, 0,
			conv_59_params_google1.pool_size, conv_59_params_google1.pool_stride, conv_59_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
   //pthread_barrier_wait(barrier_google);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[50] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
// pool_67 for Inception 5b branch 4
    start = read_cycles();
    tiled_pool_auto_cid(pool_67_params_google1.batch_size, pool_67_params_google1.out_channels, pool_67_params_google1.out_dim, pool_67_params_google1.out_dim_pooled,
		pool_67_params_google1.out_stride,
      pool_67_params_google1.pool_size, pool_67_params_google1.pool_stride, pool_67_params_google1.pool_padding,	
      (elem_t*) inception5a_out_google1, (elem_t*) pool_67_out_google1,
		orow_divide, batch_divide, cid, target_util);
gemmini_fence();
    end = read_cycles();
    total_pool_cycles += end - start;
    pool_cycles[10] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
	 // Inception 5b
	 //Branch 1
    // conv_62
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_62_params_google1.I, conv_62_params_google1.J, conv_62_params_google1.K, 1024 + 64,
        (elem_t*) inception5a_out_google1, (elem_t*)conv_62_w_google1, (acc_t*)conv_62_b_google1, (elem_t*)conv_62_out_google1,
        NO_ACTIVATION, conv_62_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[51] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
 
	 // Branch 2
    // conv_63
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_63_params_google1.I, conv_63_params_google1.J, conv_63_params_google1.K, conv_63_params_google1.out_stride,
        (elem_t*) inception5a_out_google1, (elem_t*)conv_63_w_google1, (acc_t*)conv_63_b_google1, (elem_t*)conv_63_out_google1,
        NO_ACTIVATION, conv_63_params_google1.output_scale, 0, true,
        WS,
        orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[52] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
               
    // conv_64
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_64_params_google1.batch_size, conv_64_params_google1.in_dim, conv_64_params_google1.in_channels,
			conv_64_params_google1.out_channels, conv_64_params_google1.out_dim,
			conv_64_params_google1.stride, 1, conv_64_params_google1.padding, conv_64_params_google1.kernel_size,
			1024 + 64,

			(elem_t*)conv_63_out_google1, (elem_t*)conv_64_w_google1, (acc_t*)conv_64_b_google1, (elem_t*)((elem_t*)(inception5b_out_google1) + 384),

			RELU, conv_64_params_google1.output_scale, 0,
			conv_64_params_google1.pool_size, conv_64_params_google1.pool_stride, conv_64_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[53] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    
	 // Branch 3
    // conv_65
    start = read_cycles();

	  tiled_matmul_nn_auto_cid(conv_65_params_google1.I, conv_65_params_google1.J, conv_65_params_google1.K,
			conv_65_params_google1.out_stride,
			(elem_t*) inception5a_out_google1, (elem_t*) conv_65_w_google1, (acc_t*) conv_65_b_google1, (elem_t*) conv_65_out_google1,
			RELU, conv_65_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[54] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
    // conv_66
    start = read_cycles();
	  tiled_conv_A_stride_auto_cid(
			conv_66_params_google1.batch_size, conv_66_params_google1.in_dim, conv_66_params_google1.in_channels,
			conv_66_params_google1.out_channels, conv_66_params_google1.out_dim,
			conv_66_params_google1.stride, 1, conv_66_params_google1.padding, conv_66_params_google1.kernel_size,
			1024 + 64,

			(elem_t*)conv_65_out_google1, (elem_t*)conv_66_w_google1, (acc_t*)conv_66_b_google1, (elem_t*)((elem_t*) inception5b_out_google1 + 768),

			RELU, conv_66_params_google1.output_scale, 0,
			conv_66_params_google1.pool_size, conv_66_params_google1.pool_stride, conv_66_params_google1.pool_padding, true,

			WS, orow_divide, batch_divide, cid, target_util);
   //pthread_barrier_wait(barrier_google);


    end = read_cycles();
    total_conv_cycles += end - start;
    conv_cycles[55] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
    // Branch 4
    // conv_68
    start = read_cycles();
    tiled_matmul_nn_auto_cid(conv_68_params_google1.I, conv_68_params_google1.J, conv_68_params_google1.K,
			1024 + 64,
			(elem_t*) pool_67_out_google12, (elem_t*) conv_68_w_google1, (acc_t*) conv_68_b_google1, (elem_t*) inception5b_out_google1 + 896,
			RELU, conv_68_params_google1.output_scale, 0, true,
			WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[56] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif        
       
    // Global averaging
    
    static elem_t average[1024][1] row_align(MAX_BLOCK_LEN);

    start = read_cycles();
    if(cid == 0)
       tiled_global_average_auto((elem_t*) inception5b_out_google1, (elem_t*) average, 1, 1024, 7, WS);


    end = read_cycles();
    other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif
   // fc_69
    start = read_cycles();
    tiled_matmul_nn_auto_cid(fc_69_params_google1.I, fc_69_params_google1.J, fc_69_params_google1.K, fc_69_params_google1.out_stride,
        (elem_t*)fc_69_w_google1, (elem_t*)average, (acc_t*)fc_69_b_google1, (elem_t*)fc_69_out_google1,
        NO_ACTIVATION, fc_69_params_google1.output_scale, 0, false,
        WS, orow_divide, batch_divide, cid, target_util);

    end = read_cycles();
    total_matmul_cycles += end - start;
    conv_cycles[57] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(barrier_google);
#endif   

    for(int i = 0; i < num_cycle; i++){
      if(i < 58){
        cycles[cid][i] = conv_cycles[i];
      }
      else if (i < 69){
        cycles[cid][i] = pool_cycles[i - 58];
      }
      else{
        if(i == 69) cycles[cid][i] = total_conv_cycles;
        if(i == 70) cycles[cid][i] = total_pool_cycles;
        if(i == 71) cycles[cid][i] = total_conv_cycles + total_matmul_cycles + total_pool_cycles + other_cycles;
      }
    }
    return cycles[cid];
#undef num_cycle
}
