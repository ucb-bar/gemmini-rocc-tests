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
#define SEED 2
#define total_workloads 100 // 100 each
#define QoS 0
#define WORKLOAD_CORE 2
#define QUEUE_DEPTH 5
#define NUM_ITER 4
#define CAP 4 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 0.3
#define TARGET_SCALE 0.85

#define BATCH1 true
#define BATCH4 false
#define BATCH8 false

#define debug_print 1

#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini_8.h"
#include "include/gemmini_nn.h"
#include "workload_8.h"
//pthread_barrier_t barrier[NUM_SUB_GROUP]; // between two, total 4
//pthread_barrier_t barrier_sub[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_global; // for all 8 cores


pthread_barrier_t barrier[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_sub[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_mid[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_start[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_sub_start[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_finish2[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_finish3[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_finish[NUM_SUB_GROUP]; // between two, total 4
pthread_barrier_t barrier_sub_finish[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_sub_mid2[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_sub_mid3[NUM_GROUP]; // between four, total 2
pthread_barrier_t barrier_sub_mid[NUM_GROUP]; // between four, total 2



bool done[NUM_GROUP] = {0};
int queue_group[NUM_GROUP] = {0};

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
//   int workload_num_core; 
   int workload_id;
   int cid;
   int group_id;
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
  int workload_num_core = 2;//nn_args->workload_num_core;
  int cid = nn_args->cid; // 0 or 1
  int group_id = nn_args->group_id; // 4 + 4 (0 or 1)
  done[group_id] = false;
  pthread_barrier_wait(&barrier_global);
  int total_sub_group_id = nn_args->barrier_index; // overall subgroup id 2 + 2 + 2 + 2 (0 to 3)
  int sub_group_id = (total_sub_group_id % NUM_GROUP); // inside each group's sub id 0 or 1
  int group_cid = cid + sub_group_id * SUB_GROUP; // cid inside group
  printf("entered thread_NN - cid: %d, total_sub_group_id(barrier_index): %d, sub_group_id: %d, group_id: %d, real_cid: %d\n", cid, total_sub_group_id, sub_group_id, group_id, real_cid);
  uint64_t start, end;
  while(!done[group_id]){
   pthread_barrier_wait(&barrier_sub_start[group_id]);
   if(cid == 0) {
      gemmini_done[group_id][sub_group_id] = false;
   }
   if(cid == 0 && sub_group_id == 0){
// set global time
      queue_group[group_id] = 1;
      global_time[group_id] = gemmini_runtime[0+group_id*SUB_CORE];
      for(int i = 0; i < SUB_CORE; i++) // SUB_CORE: 4
	if(global_time[group_id] < gemmini_runtime[i+group_id*SUB_CORE]) 
	   global_time[group_id] = gemmini_runtime[i+group_id*SUB_CORE]; 
// workload 
      queue_group[group_id] = workload_priority_mp(group_id, SUB_GROUP, total_workloads, NUM_ITER, global_time[group_id]);
#if debug_print == 1
       printf("group %d finished workload queue assignment, number of queue group: %d, gemmini runtime: %llu\n", group_id, queue_group[group_id], global_time[group_id]);

      for(int x = 0; x < queue_group[group_id]; x++){
         for(int y = 0; y < SUB_GROUP; y++){ 
            printf("queue %d, group %d: ", x, y);
            for(int j = 0; j < QUEUE_DEPTH; j++)
               printf("%d, ", gemmini_workload_assigned[group_id][y][x][j]);
            printf("\n");
         }
      }
#endif
       //printf("group %d finished workload queue assignment, number of queue group: %d, gemmini runtime: %llu\n", group_id, queue_group[group_id], global_time[group_id]);
 
      done[group_id] = (queue_group[group_id] == 0) && (total_queue_status[group_id][total_workloads-1] != -1);
   }
   pthread_barrier_wait(&barrier_sub_mid[group_id]);
   uint64_t temp_cycles = global_time[group_id]; //gemmini_runtime[real_cid];
   pthread_barrier_wait(&barrier_start[total_sub_group_id]);
   start = read_cycles();
   bool others_done = false;
   for(int g = 0; g < queue_group[group_id]; g++){
     for(int i = 0; i < QUEUE_DEPTH; i++){
       for(int o = 0; o < SUB_GROUP; o++){ // SUB_GROUP: 2
	 if(o != sub_group_id){
	   if(gemmini_done[group_id][o] && queue_group[group_id] == NUM_ITER) others_done = true;
	 }
       }
       bool all = false;
       int queue_id = gemmini_workload_assigned[group_id][sub_group_id][g][i];
       if(i == 0 && g == 0 && (gemmini_workload_assigned[group_id][0][0][0] == -1 ||  gemmini_workload_assigned[group_id][1][0][0] == -1)) all = true;
       //if(i == 0 && g == 0 && (gemmini_workload_assigned[group_id][0][0][0] == gemmini_workload_assigned[group_id][1][0][0])) all = true;
       if(queue_id != -1){
	if(!others_done){
          int workload_id = total_queue_type[group_id][queue_id];
	  // put score here
          if(cid == 0) {
		gemmini_score[total_sub_group_id] = (1 + total_queue_priority[group_id][queue_id]) / 4 + MAX(1, (int)(4 * (temp_cycles - total_queue_dispatch[group_id][queue_id]))/total_queue_target[group_id][queue_id]);
	
	  }
	  int group_queue_id = gemmini_workload_grouped[group_id][sub_group_id][g][i];
          pthread_barrier_wait(&barrier_mid[total_sub_group_id]); 
#if debug_print == 1
    	//printf("rid: %d, workload id: %d, queue id: %d, group queue id: %d, score: %d\n", real_cid, workload_id, queue_id, group_queue_id, gemmini_score[total_sub_group_id]);
#endif	
          uint64_t inner_start = read_cycles();
          uint64_t total_runtime = workload_function(queue_id, workload_id, cid, group_id, total_sub_group_id, all ? SUB_CORE : workload_num_core, -1, &barrier[nn_args->barrier_index]);
     //     uint64_t total_runtime = workload_function(queue_id, workload_id, all ? group_cid : cid, group_id, total_sub_group_id, all ? SUB_CORE : workload_num_core, -1, all ? &barrier_sub[group_id] : &barrier[nn_args->barrier_index]);
    
          total_queue_runtime_total[group_id][group_cid][queue_id] = total_runtime;
          uint64_t inner_end = read_cycles();
	  uint64_t this_cycles = temp_cycles + inner_end - inner_start;
          total_queue_finish[group_id][group_cid][queue_id] = (this_cycles > total_queue_dispatch[group_id][queue_id]) ? (this_cycles- total_queue_dispatch[group_id][queue_id]) : 1000;
          //total_queue_finish[group_cid][queue_id] = ((temp_cycles + inner_end - start) - total_queue_dispatch[queue_id]);

          total_queue_runtime_thread[group_id][group_cid][queue_id] = inner_end - inner_start;
	  temp_cycles += (inner_end - inner_start);
           pthread_barrier_wait(&barrier_finish[total_sub_group_id]);
 
	}
	else total_queue_status[group_id][queue_id] = -1; // release the queue 
       }
       else
          break;
     }
   }
   if(cid == 0) gemmini_done[group_id][sub_group_id] = true;
   end = read_cycles();
     pthread_barrier_wait(&barrier_sub_finish[group_id]);
   if(!done[group_id]){
     if(queue_group[group_id] == 0) gemmini_runtime[real_cid] += 1000000;
     else gemmini_runtime[real_cid] += (end - start);
     pthread_barrier_wait(&barrier_sub[group_id]);
     //if(queue_group != 1)
     //   pthread_barrier_wait(&barrier[NUM_CORE]);
     //pthread_barrier_wait(&barrier[NUM_CORE]);
     //printf("idle cycle: %llu\n", gemmini_runtime[real_cid] - temp_cycles);
   } 
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

    pthread_barrier_init(&barrier_global, NULL, NUM_CORE);
    
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub_start[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier_mid[i], NULL, WORKLOAD_CORE);
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier_start[i], NULL, WORKLOAD_CORE);
    }
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub_mid2[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub_mid3[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub_mid[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub_finish[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier_finish2[i], NULL, WORKLOAD_CORE);
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier_finish3[i], NULL, WORKLOAD_CORE);
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier_finish[i], NULL, WORKLOAD_CORE);
    }
    for(int i = 0; i < NUM_GROUP; i++){
      pthread_barrier_init(&barrier_sub[i], NULL, SUB_CORE); // between 4 cores
    }
    for(int i = 0; i < NUM_SUB_GROUP; i++){
      pthread_barrier_init(&barrier[i], NULL, WORKLOAD_CORE);
    }
    printf("starting workload creation \n");
    workload_mode_2(total_workloads+5, BATCH1, BATCH4, BATCH8, SEED, CAP, TARGET_SCALE, CAP_SCALE); 
    printf("workload creation finished \n");


    for(int i = 0; i < NUM_GROUP; i++){
      for(int j = 0; j < SUB_GROUP; j++)
	gemmini_done[i][j] = false;
      for(int j = 0; j < SUB_CORE; j++){
    	int index = i * SUB_CORE + j;
//	printf("index: %d\n", index);
    	nn_args[index].barrier_index = (int)(index / WORKLOAD_CORE);
    	nn_args[index].cid = j % WORKLOAD_CORE;
    	nn_args[index].group_id = i;
    	//pthread_create(&thread[index], &attr[index], thread_NN, &nn_args[index]);
      }
    }
    for(int i = 0; i < NUM_CORE; i++)
	pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);	
    for(int i = 0; i < NUM_CORE; i++)
  	pthread_join(thread[i], NULL);



// check total_queue_finish, total_queue_runtime_thread, total_queue_runtime_total of each workload (total_queue_type)
// also check gemmini_runtime 

  for(int group = 0; group < NUM_GROUP; group ++){
  for(int i = 0; i < total_workloads; i++){
    uint64_t max = 0;   
    for(int j = 0; j < SUB_CORE; j++){
      max = max > total_queue_finish[group][j][i] ? max : total_queue_finish[group][j][i]; 
    }
	  printf("group %d queue id %d workload type: %d\n", group, i, total_queue_type[group][i]);
	  printf("group %d queue id %d dispatch to finish time: %llu\n", group, i, max);
    
    printf("group %d queue id %d priority: %d\n", group, i, total_queue_priority[group][i]);
    printf("group %d queue id %d qos: %d\n", group, i, total_queue_qos[group][i]);
    printf("group %d queue id %d dispatched time: %llu\n", group, i, total_queue_dispatch[group][i]);
    printf("group %d queue id %d target: %llu\n", group, i, total_queue_target[group][i]);

/*

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
*/
  }
  }
  for(int i = 0; i < NUM_CORE; i++){
     printf("gemmini core id %d runtime: %llu\n", i, gemmini_runtime[i]);
  }

  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier_finish2[i]);
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier_finish3[i]);
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier_finish[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub_mid2[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub_mid3[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub_mid[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub_finish[i]);
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier_mid[i]);
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier_start[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub_start[i]);
  for(int i = 0; i < NUM_SUB_GROUP; i++)
    pthread_barrier_destroy(&barrier[i]);
  for(int i = 0; i < NUM_GROUP; i++)
    pthread_barrier_destroy(&barrier_sub[i]);
  pthread_barrier_destroy(&barrier_global); 
  printf("==================================\n");
  exit(0);
}

