#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
//#include "resnet_params_1.h"
//#include "resnet50_mt_images.h"
//#define NUM_OUTPUT (20+34+16+3)

#define NUM_CORE 4
#define SEED 100
#define total_workloads 200
#define QoS 2

#define CAP 5 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 0.8
#define TARGET_SCALE 1
#define QUEUE_DEPTH 5


#define BATCH1 true
#define BATCH4 false
#define BATCH8 false

#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "workload.h"
pthread_barrier_t barrier[NUM_CORE];

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
    uint64_t total_thread_cycles, total_cycles, total_conv_cycles, total_matmul_cycles, total_resadd_cycles;
//	uint64_t resadd_cycles[num_resadd];
//	uint64_t conv_cycles[num_layer];
//    uint64_t matmul_cycles[num_layer];
   int barrier_index;
   int workload_num_core; 
   int workload_id;
   int cid;
};
// random matmul to warm up thread
void *thread_matmul0(void *arg){
        struct thread_args * matmul_args = (struct thread_args *) arg;
        gemmini_flush(0);
        int cid = sched_getcpu();//matmul_args->i;
	printf("entered thread_matmul function - cid: %d\n", cid);
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
	    1,
            WS);
}


void *thread_NN(void *arg){
	int real_cid = sched_getcpu();
	struct thread_args * nn_args = (struct thread_args *) arg;
	gemmini_flush(0);
	uint64_t* cycles;
    int workload_num_core = nn_args->workload_num_core;
    int cid = nn_args->cid;
    

    uint64_t start, end;
    //printf("entered thread_NN - cid: %d\n", cid);
    //uint64_t target_cycle = nn_args->target_cycles;
    pthread_barrier_wait(&barrier[nn_args->barrier_index]);
    //printf("barrier working - cid: %d\n", cid);

    for(int i = 0; i < total_workloads; i++){
	int workload_id = total_queue_type[i];
    	start = read_cycles();
    	uint64_t total_runtime = workload_function(i, workload_id, cid, 0, workload_num_core, -1, &barrier[nn_args->barrier_index]);
	pthread_barrier_wait(&barrier[nn_args->barrier_index]);
    	end = read_cycles();
 
    	gemmini_runtime[real_cid] += (end - start);
	total_queue_runtime_thread[real_cid][i] = end - start;
	total_queue_runtime_total[real_cid][i] = total_runtime;
	total_queue_finish[real_cid][i] = (gemmini_runtime[real_cid] - total_queue_dispatch[i]);
    	//nn_args->total_thread_cycles = end - start;
    	//nn_args->total_cycles = total_runtime;
    }
}

void *print_message(void *ptr){
    printf("entered message thread\n");
    gemmini_flush(0); // check whether all have gemmini cores
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

    int cpu_id;
    cpu_id = sched_getcpu();
    printf("main cpu: %d \n", cpu_id);
 
    cpu_set_t cpuset[NUM_CORE];
    pthread_t thread[NUM_CORE];
    pthread_attr_t attr[NUM_CORE];
    for(int i = 0; i < NUM_CORE; i++)
	pthread_attr_init(&attr[i]);
    struct thread_args nn_args[NUM_CORE];

    printf("create threading \n");
    for(int i = 0; i < NUM_CORE; i++){
	 CPU_ZERO(&cpuset[i]);
	 CPU_SET(i, &cpuset[i]);
	 pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
	 pthread_create(&thread[i], &attr[i], print_message, NULL);
    }

    for(int i = 0; i < NUM_CORE; i++){
        pthread_join(thread[i], NULL);
    }
    printf("thread joined after message printing\n");

    //just random turn
    for(int i = 0; i < NUM_CORE; i++){
        pthread_create(&thread[i], &attr[i], thread_matmul0, &nn_args[i]);
    }

    for(int i = 0; i < NUM_CORE; i++)
        pthread_join(thread[i], NULL);
    printf("thread joined after doing thread matmul\n");

   // for(int i = 0; i < OROW_DIVIDE; i++)
   //     nn_args[i].target_cycles = RESNET_TARGET;
    
    //pthread_barrier_init(&barrier, NULL, OROW_DIVIDE);
    pthread_barrier_init(&barrier[0], NULL, NUM_CORE);
    printf("starting workload crreation \n");
    workload_mode_2(total_workloads, BATCH1, BATCH4, BATCH8, SEED, CAP, TARGET_SCALE, CAP_SCALE); 
    printf("workload creation finished \n");
   
    for(int i = 0; i < NUM_CORE; i++){
	nn_args[i].barrier_index = 0; // all 0
	nn_args[i].workload_num_core = NUM_CORE;
	nn_args[i].cid = i;
	pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);
    }	
    for(int i = 0; i < NUM_CORE; i++)
	pthread_join(thread[i], NULL);

// check total_queue_finish, total_queue_runtime_thread, total_queue_runtime_total of each workload (total_queue_type)
// also check gemmini_runtime 

     for(int i = 0; i < total_workloads; i++){
	uint64_t max = 0;   
	for(int j = 0; j < NUM_CORE; j++){
	   max = max > total_queue_finish[j][i] ? max : total_queue_finish[j][i]; 
	}
	printf("queue id %d workload type: %d\n", i, total_queue_type[i]);
	printf("queue id %d dispatch to finish time: %llu\n", i, max);

	printf("queue id %d priority: %d\n", i, total_queue_priority[i]);
	printf("queue id %d qos: %d\n", i, total_queue_qos[i]);
	printf("queue id %d dispatched time: %llu\n", i, total_queue_dispatch[i]);
	printf("queue id %d target: %llu\n", i, total_queue_target[i]);

	max = 0;
	for(int j = 0; j < NUM_CORE; j++){
	   max = max > total_queue_runtime_thread[j][i] ? max : total_queue_runtime_thread[j][i]; 
	}
	printf("queue id %d thread runtime: %llu\n", i, max);

	max = 0;
	for(int j = 0; j < NUM_CORE; j++){
	   max = max > total_queue_runtime_total[j][i] ? max : total_queue_runtime_total[j][i]; 
	}
	printf("queue id %d total runtime: %llu\n", i, max);
     }

     for(int i = 0; i < NUM_CORE; i++){
	printf("gemmini core id %d runtime: %llu\n", i, gemmini_runtime[i]);
     }	
    pthread_barrier_destroy(&barrier[0]);
 
    printf("==================================\n");
    exit(0);
}

