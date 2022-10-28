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
#define NUM_CORE 4
#define SEED 10
#define total_workloads 200
#define QoS 0
#define WORKLOAD_CORE 2
#define NUM_GROUP (int)(NUM_CORE / WORKLOAD_CORE)
#define QUEUE_DEPTH 5
#define NUM_ITER 5
#define CAP 5 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 0.5
#define TARGET_SCALE 1

#define BATCH1 false
#define BATCH4 false
#define BATCH8 true


#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "workload.h"
pthread_barrier_t barrier[NUM_CORE+1];


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
    int queue_group = nn_args->queue_group;
    int group_id = nn_args->barrier_index;
    uint64_t start, end;
    //printf("entered thread_NN - cid: %d, group_id: %d, real_cid: %d, initial assign: %d\n", cid, group_id, real_cid, gemmini_workload_assigned[group_id]);
    pthread_barrier_wait(&barrier[nn_args->barrier_index]);
    //printf("barrier working - group_id: %d\n", group_id);
    for(int g = 0; g < queue_group; g++){
      if(cid == 0 && group_id == 0){
        global_time = gemmini_runtime[0];
        for(int i = 0; i < NUM_CORE; i++)
	  if(global_time < gemmini_runtime[i]) 
	     global_time = gemmini_runtime[i];
	if(global_time < total_queue_dispatch[gemmini_workload_assigned[1][g][0]] ||  global_time < total_queue_dispatch[gemmini_workload_assigned[0][g][0]]){
	  global_time = MAX(total_queue_dispatch[gemmini_workload_assigned[0][g][0]], total_queue_dispatch[gemmini_workload_assigned[1][g][0]]) + 1000000;
	  gemmini_runtime[0] = global_time;
	}
      }
      pthread_barrier_wait(&barrier[NUM_CORE]);
       start = read_cycles();
       uint64_t temp_cycles = global_time;//gemmini_runtime[real_cid];
       for(int i = 0; i < QUEUE_DEPTH; i++){
	   int queue_id = gemmini_workload_assigned[group_id][g][i];
	   if(queue_id != -1){
	      int workload_id = total_queue_type[queue_id];
	//printf("rid: %d, workload id: %d, queue id: %d\n", real_cid, workload_id, queue_id);
	      uint64_t inner_start = read_cycles();
	      uint64_t total_runtime = workload_function(queue_id, workload_id, cid, group_id, workload_num_core, -1,  &barrier[nn_args->barrier_index]);
	      pthread_barrier_wait(&barrier[nn_args->barrier_index]);
	      uint64_t inner_end = read_cycles();
	 
	      temp_cycles += (inner_end - inner_start);
	      total_queue_runtime_thread[real_cid][queue_id] = inner_end - inner_start;
	      total_queue_runtime_total[real_cid][queue_id] = total_runtime;
	      total_queue_finish[real_cid][queue_id] = (temp_cycles - total_queue_dispatch[queue_id]);
	 
	   }
	   else
	      break;
       }
	pthread_barrier_wait(&barrier[NUM_CORE]);
	end = read_cycles();
	gemmini_runtime[real_cid] += (end - start);
	//printf("idle cycle for queue group %d: %llu\n", g, gemmini_runtime[real_cid] - temp_cycles);
    }
}

/*
void *thread_master(void *arg){
  bool idle_detected = false;
  int group_id = -1;
  smallest_pointer = 0;
  gemmini_workload_assigned[0] = 0;
  gemmini_workload_assigned[1] = 1;
  smallest_pointer = 1;
  printf("core id of master thread: %d\n", sched_getcpu());
  while(smallest_pointer != total_workloads - 1){
    smallest_pointer += 1;
    while(!idle_detected){
      for(int i = 0; i < NUM_GROUP; i++){
        if(gemmini_workload_assigned[i] == -1){
          idle_detected = true;
          group_id = i;
          break;
        }
      }
    } 
    gemmini_workload_assigned[group_id] = smallest_pointer; // FCFS
    idle_detected = false;
printf("group id: %d, smallest pointer: %d\n", group_id, smallest_pointer);
  }
  int count = 0;
  while(count != WORKLOAD_CORE){ 
    for(int i = 0; i < NUM_CORE; i++){
      if(gemmini_workload_received[i] == total_workloads - 1){
        count ++;
      }
    }
  }
  smallest_pointer = total_workloads;

}
*/
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
    
    pthread_barrier_init(&barrier[NUM_CORE], NULL, NUM_CORE);
    int group = NUM_CORE / WORKLOAD_CORE;
    for(int i = 0; i < group; i++){
      pthread_barrier_init(&barrier[i], NULL, WORKLOAD_CORE);
    }
    printf("starting workload crreation \n");
    workload_mode_1(QoS, total_workloads, BATCH1, BATCH4, BATCH8, SEED, CAP, TARGET_SCALE, CAP_SCALE); 

    printf("workload creation finished \n");
    int queue_group = workload_fcfs_mp_schedule(NUM_GROUP, total_workloads);
    printf("finished workload queue assignment\n");
 
    for(int i = 0; i < group; i++){
      for(int j = 0; j < WORKLOAD_CORE; j++){
        int index = i * WORKLOAD_CORE + j;
        nn_args[index].barrier_index = i;
        nn_args[index].workload_num_core = WORKLOAD_CORE;
        nn_args[index].cid = j;
	nn_args[index].queue_group = queue_group;
        pthread_create(&thread[index], &attr[index], thread_NN, &nn_args[index]);
      }
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

  for(int i = 0; i < group; i++)
    pthread_barrier_destroy(&barrier[i]);
  pthread_barrier_destroy(&barrier[NUM_CORE]); 
  printf("==================================\n");
  exit(0);
}
