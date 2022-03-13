#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "alexnet_orow_params.h"
#include "resnet50_mt_images.h"

#define num_proc 4
#define num_layer 9
#define num_resadd 1

#define THREAD_SYNC true // must do sync
#define BATCH_DIVIDE 1
#define OROW_DIVIDE 4 // 1: independent, 2: 2+2 collab, 4: sequential

#define A_no_max_block 0
#define B_no_max_block 0

#define priority false // ToDo: set it to true for priorized cores
#define target_util 0 // ToDo: needs to be changed for target utilization
#define bubble 0
#define target_util_fc 0

#define ALEXNET_REPEAT 8

pthread_barrier_t barrier;

#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS true

//meaningless
static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t Out[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

struct thread_args{
    uint64_t total_thread_cycles, total_conv_cycles, total_matmul_cycles, total_resadd_cycles, total_pool_cycles;
	uint64_t res_add_cycles[num_resadd];
	uint64_t conv_cycles[num_layer];
	uint64_t pool_cycles[num_layer];
    uint64_t matmul_cycles[num_layer];
	uint64_t other_cycles; //global average
};
// random matmul to warm up thread
void *thread_matmul0(void *arg){
        struct thread_args * matmul_args = (struct thread_args *) arg;
        gemmini_flush(0);
        int cid = sched_getcpu();//matmul_args->i;
          elem_t* A = (elem_t*) in_A + MAT_DIM_K*(MAT_DIM_I/2)*(cid/2);
          elem_t* B = (elem_t*) in_B + (MAT_DIM_J/2)*(cid%2);
          elem_t* C = (elem_t*) Out + (MAT_DIM_J/2)*(cid%2) + MAT_DIM_J*(MAT_DIM_I/2)*(cid/2);
	if(cid == 0 || cid == 1)
          tiled_matmul_auto(MAT_DIM_I/2, MAT_DIM_J/2, MAT_DIM_K,
                                A, B, NULL, C, //NO_BIAS ? NULL : D, C,
                           MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            false, false,
            false, !FULL_BIAS_WIDTH,
            WS);
}


void *thread_NN(void *arg){
	int cid = sched_getcpu();
	struct thread_args * nn_args = (struct thread_args *) arg;
    enum tiled_matmul_type_t tiled_matmul_type = WS;
	gemmini_flush(0);
    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;
    //int image_offset = conv_1_params.in_channels * conv_1_params.in_dim * conv_1_params.in_dim * cid;
    pthread_barrier_wait(&barrier);
    
    uint64_t thread_start = read_cycles();
    

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

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
 
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_1_params.batch_size,
        conv_1_params.out_channels, conv_1_params.out_dim, conv_1_params.out_dim_pooled,
        conv_1_params.out_stride,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

        (elem_t*)conv_1_out, (elem_t*)conv_1_out_pooled,
	OROW_DIVIDE, BATCH_DIVIDE, cid);

    end = read_cycles();
    pool_cycles += end - start;
    nn_args->pool_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
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

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
  
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_2_params.batch_size,
        conv_2_params.out_channels, conv_2_params.out_dim, conv_2_params.out_dim_pooled,
        conv_2_params.out_stride,
        conv_2_params.pool_size, conv_2_params.pool_stride, conv_2_params.pool_padding,

        (elem_t*)conv_2_out, (elem_t*)conv_2_out_pooled,
	OROW_DIVIDE, BATCH_DIVIDE, cid);

    end = read_cycles();
    pool_cycles += end - start;
    nn_args->pool_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
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

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
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

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[3] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
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

        WS, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util, bubble);

    end = read_cycles();
    conv_cycles += end - start;
    nn_args->conv_cycles[4] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif        
     
    start = read_cycles();
    tiled_pool_auto_cid(
        conv_5_params.batch_size,
        conv_5_params.out_channels, conv_5_params.out_dim, conv_5_params.out_dim_pooled,
        conv_5_params.out_stride,
        conv_5_params.pool_size, conv_5_params.pool_stride, conv_5_params.pool_padding,

        (elem_t*)conv_5_out, (elem_t*)conv_5_out_pooled,
	OROW_DIVIDE, BATCH_DIVIDE, cid);

    end = read_cycles();
    pool_cycles += end - start;
    nn_args->pool_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif           
    // Global averaging
    
    static elem_t average[1][9216] row_align(MAX_BLOCK_LEN);

    start = read_cycles();
    if(cid == 0)
        tiled_global_average_auto(conv_5_out_pooled, average, conv_5_params.batch_size,                         
            conv_5_params.out_channels, conv_5_params.out_dim, WS);
       

    end = read_cycles();
    nn_args->other_cycles = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif

    // fc_6
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_6_params.I, fc_6_params.J, fc_6_params.K, fc_6_params.out_stride,
        (elem_t*)average, (elem_t*)fc_6_w, (acc_t*)fc_6_b, (elem_t*)fc_6_out,
        RELU, fc_6_params.output_scale, 0, false,
        tiled_matmul_type, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_fc, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[0] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif   

    // fc_7
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_7_params.I, fc_7_params.J, fc_7_params.K, fc_7_params.out_stride,
        (elem_t*)fc_6_out, (elem_t*)fc_7_w, (acc_t*)fc_7_b, (elem_t*)fc_7_out,
        RELU, fc_7_params.output_scale, 0, false,
        tiled_matmul_type, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_fc, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[1] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif   

    // fc_8
    start = read_cycles();

    tiled_matmul_nn_auto_cid(fc_8_params.I, fc_8_params.J, fc_8_params.K, fc_8_params.out_stride,
        (elem_t*)fc_7_out, (elem_t*)fc_8_w, (acc_t*)fc_8_b, (elem_t*)fc_8_out,
        NO_ACTIVATION, fc_8_params.output_scale, 0, false,
        tiled_matmul_type, OROW_DIVIDE, BATCH_DIVIDE, cid, A_no_max_block, B_no_max_block, priority, target_util_fc, bubble);

    end = read_cycles();
    matmul_cycles += end - start;
    nn_args->matmul_cycles[2] = end - start;
#if THREAD_SYNC == 1
    pthread_barrier_wait(&barrier);
#endif   


    uint64_t thread_end = read_cycles();
    nn_args->total_thread_cycles = thread_end - thread_start;
    nn_args->total_matmul_cycles = matmul_cycles;
    nn_args->total_conv_cycles = conv_cycles;
    nn_args->other_cycles = other_cycles;
    nn_args->total_pool_cycles = pool_cycles;
    nn_args->total_resadd_cycles = res_add_cycles;

}
void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
    printf("print msg - cpu_id: %d \n", cpu_id);
}

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    bool conv = true;
    bool check = false;
    int cpu_id;
    cpu_id = sched_getcpu();
    cpu_set_t cpuset[num_proc];
    pthread_t thread[num_proc];
    pthread_attr_t attr[num_proc];
    for(int i = 0; i < num_proc; i++)
	pthread_attr_init(&attr[i]);
    struct thread_args nn_args[num_proc];


    for(int i = 0; i < num_proc; i++){
	 CPU_ZERO(&cpuset[i]);
	 CPU_SET(i, &cpuset[i]);
	 pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
	 pthread_create(&thread[i], &attr[i], print_message, NULL);
    }

    for(int i = 0; i < num_proc; i++){
        pthread_join(thread[i], NULL);
    }

    //just random turn
    for(int i = 0; i < num_proc; i++){
        pthread_create(&thread[i], &attr[i], thread_matmul0, &nn_args[i]);
    }

    for(int i = 0; i < num_proc; i++)
        pthread_join(thread[i], NULL);

    pthread_barrier_init(&barrier, NULL, OROW_DIVIDE);
    
    for(int r = 0; r < ALEXNET_REPEAT; r++){
        uint64_t start = read_cycles();
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
        
        for(int i = 0; i < OROW_DIVIDE; i++)
            pthread_join(thread[i], NULL);
        uint64_t end = read_cycles();
        
        printf("alexnet repeat %d total cycles with threading overhead: %llu \n", r, end - start);


        uint64_t thread_resnet_max = 0;
        uint64_t total_resnet_max = 0;
        for(int i = 0; i < OROW_DIVIDE; i++){
            uint64_t matmul_cycles = nn_args[i].total_matmul_cycles;
            uint64_t conv_cycles = nn_args[i].total_conv_cycles;
	    uint64_t other_cycles = nn_args[i].other_cycles;
	    uint64_t pool_cycles = nn_args[i].total_pool_cycles;
            uint64_t total_cycles =  conv_cycles + matmul_cycles + pool_cycles + other_cycles;
            uint64_t thread_cycles = nn_args[i].total_thread_cycles;
            thread_resnet_max = thread_resnet_max > thread_cycles ? thread_resnet_max : thread_cycles;
            total_resnet_max = total_resnet_max > total_cycles ? total_resnet_max : total_cycles;
        }
        printf("\nalexnet repeat %d total thread cycles: %llu\n", r, thread_resnet_max);
        printf("alexnet repeat %d total cycles: %llu\n", r, total_resnet_max);


        printf("worst case for each layers \n");
        

        for(int i = 0; i < 5; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].conv_cycles[i]) ? max : nn_args[j].conv_cycles[i];
            
            printf("alexnet repeat %d Conv layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        }
        

        for(int i = 0; i < 3; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].matmul_cycles[i]) ? max : nn_args[j].matmul_cycles[i];
            
            printf("alexnet repeat %d Matmul layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        

        }

        for(int i = 0; i < 3; i++)

        {
            uint64_t max = 0;
            for(int j = 0; j < OROW_DIVIDE; j++)
               max = (max > nn_args[j].pool_cycles[i]) ? max : nn_args[j].pool_cycles[i];
            
            printf("alexnet repeat %d Pool layer %d worst cycles: %llu \n", r, i, max);
            max = 0;
        

        }
        
    }
    printf("==================================\n");
    
    pthread_barrier_destroy(&barrier);
    
    exit(0);
}

