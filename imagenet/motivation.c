// split to 2 cores each, multi-program but first-come-first-served
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#define NUM_CORE 8
#define SEED 0
#define QoS 0
#define WORKLOAD_CORE 2 // 2 core or 4 core
#define NUM_GROUP (int)(NUM_CORE / WORKLOAD_CORE)
#define total_workloads 200 // per group 

#define BATCH1 true
#define BATCH4 false
#define BATCH8 false

#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "motivation.h"
pthread_barrier_t barrier[NUM_GROUP];
pthread_barrier_t barrier_workload[NUM_GROUP];
pthread_barrier_t barrier_global;

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
   int cid;
   int queue_group;
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
  int group = nn_args->barrier_index;
  uint64_t start, end;
  //printf("entered thread_NN - cid: %d, group_id: %d, real_cid: %d, initial assign: %d\n", cid, group_id, real_cid, gemmini_workload_assigned[group_id]);
  pthread_barrier_wait(&barrier_global);

  for(int i = 0; i < total_workloads; i ++){
    pthread_barrier_wait(&barrier[group]);
    int workload_type = total_queue_type[group][i];
    start = read_cycles();
    uint64_t total_runtime = workload_function(workload_type, cid, group, workload_num_core, &barrier_workload[group]);
    end = read_cycles();

    pthread_barrier_wait(&barrier[group]);
    total_queue_runtime_thread[group][cid][i] = inner_end - inner_start;
	  total_queue_runtime_total[group][cid][i] = total_runtime;
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
    
    pthread_barrier_init(&barrier_global, NULL, NUM_CORE);
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier[i], NULL, WORKLOAD_CORE);
      pthread_barrier_init(&barrier_workload[i], NULL, WORKLOAD_CORE);
    }
    printf("starting workload crreation \n");
    workload_create(total_workloads, BATCH1, BATCH4, BATCH8); 
    printf("workload creation finished \n");

    for(int i = 0; i < NUM_GROUP; i++){
      for(int j = 0; j < WORKLOAD_CORE; j++){
        int index = i * WORKLOAD_CORE + j;
        nn_args[index].barrier_index = i;
        nn_args[index].workload_num_core = WORKLOAD_CORE;
        nn_args[index].cid = j;
        pthread_create(&thread[index], &attr[index], thread_NN, &nn_args[index]);
      }
    }	
    for(int i = 0; i < NUM_CORE; i++)
      pthread_join(thread[i], NULL);

// check total_queue_finish, total_queue_runtime_thread, total_queue_runtime_total of each workload (total_queue_type)
// also check gemmini_runtime 

  for(int g = 0; g < NUM_GROUP; g++){
    for(int i = 0; i < total_workloads; i++){
      uint64_t max = 0;   
      int index = i + g * total_workloads;
      printf("queue id %d workload type: %d\n", index, total_queue_type[g][i]);
      
      
     // printf("queue id %d target: %llu\n", i, total_queue_target[i]);


      max = 0;
      for(int j = 0; j < WORKLOAD_CORE; j++){
        max = max > total_queue_runtime_thread[g][j][i] ? max : total_queue_runtime_thread[g][j][i]; 
      }
      printf("queue id %d thread runtime: %llu\n", index, max);

      max = 0;
      for(int j = 0; j < WORKLOAD_CORE; j++){
         max = max > total_queue_runtime_total[g][j][i] ? max : total_queue_runtime_total[g][j][i]; 
      }
      printf("queue id %d total runtime: %llu\n", index, max);
    }
  }

  for(int i = 0; i < group; i++){
    pthread_barrier_destroy(&barrier_workload[i]);
    pthread_barrier_destroy(&barrier[i]);
  }
  pthread_barrier_destroy(&barrier_global); 
  printf("==================================\n");
  exit(0);
}

