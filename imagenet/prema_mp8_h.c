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
//#include "resnet_params_1.h"
//#include "resnet50_mt_images.h"
//#define NUM_OUTPUT (20+34+16+3)

#define NUM_CORE 8
#define SUB_CORE 8
#define SEED 2
#define total_workloads 230 // 100 each
#define WORKLOAD_CORE 2
#define QUEUE_DEPTH 5
#define NUM_ITER 5
#define CAP 4 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 1.55
#define TARGET_SCALE 0.85

#define BATCH1 true
#define BATCH4 false
#define BATCH8 false

#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "workload_8.h"
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
   int workload_id;
   int cid;
   int queue_length;
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
  int cid = nn_args->cid;
  int queue_length = nn_args->queue_length;
  uint64_t start, end;
  int workload_num_core = NUM_CORE;
  uint64_t temp_cycles = global_time; //gemmini_runtime[real_cid];
  //printf("entered thread_NN - cid: %d, group_id: %d, real_cid: %d, initial assign: %d\n", cid, group_id, real_cid, gemmini_workload_assigned[group_id]);
  pthread_barrier_wait(&barrier_global);
  //pthread_barrier_wait(&barrier[NUM_CORE]);
  //printf("barrier working - group_id: %d\n", group_id);
  start = read_cycles();
  for(int i = 0; i < QUEUE_DEPTH; i++){
     int queue_id = gemmini_workload_assigned[0][0][0][i];
     if(queue_id != -1){
        uint64_t inner_start = read_cycles();

        int workload_id = total_queue_type[queue_id];

    //printf("rid: %d, workload id: %d, queue id: %d\n", real_cid, workload_id, queue_id);
        uint64_t total_runtime = workload_function(queue_id, workload_id, cid, 0, 0, workload_num_core, -1,  &barrier_global);
//          pthread_barrier_wait(&barrier_global);
     
        //temp_cycles += (inner_end - inner_start);
        total_queue_runtime_total[real_cid][queue_id] = total_runtime;
        uint64_t inner_end = read_cycles();
        total_queue_runtime_thread[real_cid][queue_id] = inner_end - inner_start;
        uint64_t this_cycles = temp_cycles + inner_end - start;
        total_queue_finish[real_cid][queue_id] = (this_cycles > total_queue_dispatch[queue_id]) ? (this_cycles- total_queue_dispatch[queue_id]) : 1000;
        pthread_barrier_wait(&barrier_global);
   
        //total_queue_finish[real_cid][queue_id] = ((temp_cycles + inner_end - start) - total_queue_dispatch[queue_id]);
     }
     else
       break;
  }
  end = read_cycles();
  pthread_barrier_wait(&barrier_global);
  //if(queue_group != 1)
  //   pthread_barrier_wait(&barrier[NUM_CORE]);
  //pthread_barrier_wait(&barrier[NUM_CORE]);
  gemmini_runtime[real_cid] += (end - start);
  //printf("idle cycle: %llu\n", gemmini_runtime[real_cid] - temp_cycles);
    
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
/*
    int group = NUM_CORE / WORKLOAD_CORE;
    for(int i = 0; i < group; i++){
      pthread_barrier_init(&barrier[i], NULL, WORKLOAD_CORE);
    }
    */
    printf("starting workload creation \n");
    workload_mode_2(total_workloads, BATCH1, BATCH4, BATCH8, SEED, TARGET_SCALE, CAP_SCALE); 
    printf("workload creation finished \n");

    int queue_length = 1;
    while(queue_length != -1){
      global_time = gemmini_runtime[0];
      for(int i = 0; i < NUM_CORE; i++)
        if(global_time < gemmini_runtime[i]) 
          global_time = gemmini_runtime[i];
      queue_length = workload_priority_sp(total_workloads, global_time); // or instead use max cycle
      //workload_lengthing(queue_length, NUM_GROUP);
      printf("finished workload queue assignment, number of queue length: %d, gemmini runtime: %d\n", queue_length, global_time);

      for(int j = 0; j < QUEUE_DEPTH; j++)
        printf("%d, ", gemmini_workload_assigned[0][0][0][j]);
      printf("\n");
      
      
      if(queue_length > 0){
        for(int j = 0; j < NUM_CORE; j++){
          //gemmini_done[j] = false;
          //nn_args[index].barrier_index = i;
          nn_args[j].workload_num_core = WORKLOAD_CORE;
          nn_args[j].cid = j;
          nn_args[j].queue_length = queue_length;
          pthread_create(&thread[j], &attr[j], thread_NN, &nn_args[j]);
        }
        for(int i = 0; i < NUM_CORE; i++)
          pthread_join(thread[i], NULL);
      }
      else{
        for(int i = 0; i < NUM_CORE; i++)
          gemmini_runtime[i] += 100000;
      }	
    }

  for(int i = 0; i < total_workloads; i++){
    uint64_t max = 0;     
    for(int j = 0; j < SUB_CORE; j++){
      max = max > total_queue_finish[j][i] ? max : total_queue_finish[j][i]; 
    }
	  printf("queue id %d workload type: %d\n", i, total_queue_type[i]);
	  printf("queue id %d dispatch to finish time: %llu\n", i, max); 
    printf("queue id %d priority: %d\n", i, total_queue_priority[i]);
    printf("queue id %d dispatched time: %llu\n", i, total_queue_dispatch[i]);
    printf("queue id %d target: %llu\n", i, total_queue_target[i]);

    max = 0;
    for(int j = 0; j < SUB_CORE; j++){
      max = max > total_queue_runtime_thread[j][i] ? max : total_queue_runtime_thread[j][i]; 
    }
    printf("queue id %d thread runtime: %llu\n", i, max);

    max = 0;
    for(int j = 0; j < SUB_CORE; j++){
       max = max > total_queue_runtime_total[j][i] ? max : total_queue_runtime_total[j][i]; 
    }
    printf("queue id %d total runtime: %llu\n", i, max);
  }

  for(int i = 0; i < NUM_CORE; i++){
	  printf("gemmini core id %d runtime: %llu\n", i, gemmini_runtime[i]);
  }


  pthread_barrier_destroy(&barrier_global); 
  printf("==================================\n");
  exit(0);
}

