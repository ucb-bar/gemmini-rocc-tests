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
#define total_workloads 180 // 100 each
#define WORKLOAD_CORE 2
#define QUEUE_DEPTH 5
#define NUM_ITER 4
#define CAP 4 // 0 to 1 (smaller number: shorter time between workload dispatch time)
#define CAP_SCALE 1.9
#define TARGET_SCALE 0.85

#define BATCH1 true
#define BATCH4 false
#define BATCH8 false

#define debug_print 0

#define num_layer 54
#define num_resadd 16
#define num_proc NUM_CORE

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#define planaria_scale 2.5
static uint64_t gemmini_planaria_score[NUM_SUB_GROUP] = {0};
#include "workload_8.h"
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


pthread_barrier_t barrier_global_mid; // for all 8 cores


pthread_barrier_t barrier_global_start; // for all 8 cores


pthread_barrier_t barrier_global_end; // for all 8 cores


pthread_barrier_t barrier_global; // for all 8 cores


static int queue_group = 1;
bool done[NUM_GROUP] = {0};
static int curr_queue_id[NUM_SUB_GROUP] = {0};
static int total_queue_planaria_send[MAX_WORKLOAD] = {0};
static int total_queue_planaria_receive[MAX_WORKLOAD] = {0};
static uint64_t total_queue_planaria_time[MAX_WORKLOAD] = {0};
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
   int group_id;
  // int queue_group;
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
printf("global: %d, global start: %d, global end: %d, global mid: %d\n", &barrier_global, &barrier_global_start, &barrier_global_end, &barrier_global_mid);
  uint64_t* cycles;
  int workload_num_core = WORKLOAD_CORE;//nn_args->workload_num_core;
  int cid = nn_args->cid; // 0 or 1
  int group_id = nn_args->group_id; // 4 + 4 (0 or 1)
  done[group_id] = false;
  pthread_barrier_wait(&barrier_global);
  int total_sub_group_id = nn_args->barrier_index; // overall subgroup id 2 + 2 + 2 + 2 (0 to 3)
  int sub_group_id = (total_sub_group_id % NUM_GROUP); // inside each group's sub id 0 or 1
  int group_cid = cid + sub_group_id * SUB_GROUP; // cid inside group
  int other_total_sub_group_id = group_id * WORKLOAD_CORE + (total_sub_group_id % 2 == 0 ? 1 : 0);
//  printf("entered thread_NN - cid: %d, total_sub_group_id(barrier_index): %d, sub_group_id: %d, group_id: %d, real_cid: %d\n", cid, total_sub_group_id, sub_group_id, group_id, real_cid);
  uint64_t start, end;
  bool all = false; // ToDo


  while((queue_group != 0) || (total_queue_status[total_workloads-1] == -1)){
    pthread_barrier_wait(&barrier_global_start);
    if(real_cid != 0) printf("start queue assignment rid: %d\n", real_cid);
//printf("global: %d, global start: %d, global end: %d, global mid: %d\n", &barrier_global, &barrier_global_start, &barrier_global_end, &barrier_global_mid);
    //if(cid == 0) {
    //  gemmini_done[total_sub_group_id] = false;
    //}
    if(real_cid == 0){
      global_time = gemmini_runtime[0];
      for(int i = 0; i < NUM_CORE; i++)
        if(global_time < gemmini_runtime[i]) 
          global_time = gemmini_runtime[i];
      queue_group = workload_priority_mp(total_workloads, NUM_ITER, global_time); // or instead use max cycle
      //workload_grouping(queue_group, NUM_GROUP);
      printf("finished workload queue assignment, number of queue group: %d, gemmini runtime: %d\n", queue_group, global_time);

      for(int j = 0; j < NUM_SUB_GROUP; j++)
        gemmini_done[j] = false;
      for(int k = 0; k < NUM_GROUP; k++)
        for(int x = 0; x < queue_group; x++){
           for(int y = 0; y < SUB_GROUP; y++){ 
              printf("group %d queue %d, sub-group %d: ", k, x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++)
                 printf("%d, ", gemmini_workload_assigned[k][y][x][j]);
              printf("\n");
           }
        }
    }
    if(real_cid != 0) printf("waitiing rid %d\n", real_cid);

   /////////////////////////// finished creating workloads ////////////////////////////

  pthread_barrier_wait(&barrier_global);

//  printf("before mid - rid %d group %d finished workload queue assignment, number of queue group: %d, gemmini runtime: %llu, gemmini done: %d\n", real_cid, group_id, queue_group, global_time, gemmini_done[other_total_sub_group_id]);
//  pthread_barrier_wait(&barrier_global_mid);

  start = read_cycles();
  int workload_num = 0;
   for(int x = 0; x < queue_group; x++)
      for(int j = 0; j < QUEUE_DEPTH; j++)
        if(gemmini_workload_assigned[group_id][sub_group_id][x][j] >= 0) workload_num ++;
//printf("after mid - global: %d, global start: %d, global end: %d, global mid: %d\n", &barrier_global, &barrier_global_start, &barrier_global_end, &barrier_global_mid); 
//    printf("after mid - rid %d group %d finished workload queue assignment, number of queue group: %d, number of workload: %d, gemmini runtime: %llu, gemmini done: %d\n", real_cid, group_id, queue_group, workload_num, global_time,  gemmini_done[other_total_sub_group_id]);

   pthread_barrier_wait(&barrier_sub_start[group_id]);
 
   uint64_t temp_cycles = global_time; //gemmini_runtime[real_cid];
   bool others_done = false;
   for(int g = 0; g < queue_group; g++){
     for(int i = 0; i < QUEUE_DEPTH; i++){
        pthread_barrier_wait(&barrier_start[total_sub_group_id]);
//       if(cid == 0 && workload_num == 1) gemmini_last[sub_group_id] = true;
       for(int o = 0; o < NUM_SUB_GROUP; o++){ // SUB_GROUP: 2
         if(o != total_sub_group_id&& (i != 0 && g != 0)){
         //if(o != total_sub_group_id){
           if(gemmini_done[o] && queue_group == NUM_ITER ) others_done = true;
         }
       }
       bool all = false;
       int queue_id = gemmini_workload_assigned[group_id][sub_group_id][g][i];
  //     if(i == 0 && g == 0 && (gemmini_workload_assigned[group_id][0][0][0] == -1 ||  gemmini_workload_assigned[group_id][1][0][0] == -1)) all = true;
       curr_queue_id[total_sub_group_id] = queue_id; 
       int workload_id = total_queue_type[queue_id];
#if debug_print == 1
       printf("rid: %d, workload id: %d, queue id: %d,  status: %d, others done: %d, all: %d, queue_group: %d, gemmini done: %d\n", real_cid, workload_id, queue_id, total_queue_status[queue_id], others_done, all, queue_group, gemmini_done[total_sub_group_id]);
      //fflush();
#endif
          uint64_t inner_start = read_cycles();
       pthread_barrier_wait(&barrier_mid[total_sub_group_id]);
       if(queue_id != -1){
         workload_num --;
         if(!others_done){
	  // put score here
          uint64_t total_runtime = 0;
          uint64_t after_dispatch = (temp_cycles > total_queue_dispatch[queue_id]) ? (temp_cycles - total_queue_dispatch[queue_id]) : 0;
          uint64_t slack = total_queue_target[queue_id] > after_dispatch ? total_queue_target[queue_id] - after_dispatch : 10;
          if(cid == 0) {
            //gemmini_planaria_score[total_sub_group_id] = (1 + total_queue_priority[group_id][queue_id]) / 4 + MAX(1, (int)(4 * (temp_cycles - total_queue_dispatch[group_id][queue_id]))/total_queue_target[group_id][queue_id]);
            gemmini_planaria_score[total_sub_group_id] = total_queue_priority[queue_id]*10000 + ((CAP*10000*after_dispatch) / (total_queue_target[queue_id]));
 	    //gemmini_planaria_score[i] = ((1+total_queue_priority[queue_id])*100000000) / slack; 
          }
          pthread_barrier_wait(&barrier_start[total_sub_group_id]);
 //         int group_queue_id = gemmini_workload_grouped[group_id][sub_group_id][g][i];
#if debug_print == 1
    	printf("rid: %d, workload id: %d, queue id: %d,  status: %d, score: %llu, others score: %llu\n", real_cid, workload_id, queue_id, total_queue_status[queue_id], gemmini_planaria_score[total_sub_group_id], gemmini_planaria_score[other_total_sub_group_id]);

#endif	
          while(total_queue_status[queue_id] < planaria_group[workload_id]){
            uint64_t temp_end = read_cycles();
            bool inner_done = false;
	    uint64_t slack_time = slack > (temp_end - inner_start) ? slack - (temp_end - inner_start) : 1000;
            //uint64_t slack_time = (temp_cycles > (temp_end - inner_start + total_queue_dispatch[queue_id])) ? temp_cycles - (temp_end - inner_start) - total_queue_dispatch[queue_id] : 100000;
            if(workload_num == 0 || queue_group < NUM_ITER) // for last one
		total_runtime = workload_function(queue_id, workload_id, cid, group_id, total_sub_group_id, workload_num_core, slack_time, &barrier[total_sub_group_id]);
            else
              total_runtime += workload_planaria_function(queue_id, workload_id, cid, group_id, all ? total_sub_group_id / SUB_GROUP : total_sub_group_id, all ? SUB_CORE : workload_num_core, slack_time, &barrier[total_sub_group_id]);
 //            total_runtime += workload_planaria_function(queue_id, workload_id, all ? group_cid : cid, group_id, all ? total_sub_group_id / SUB_GROUP : total_sub_group_id, all ? SUB_CORE : workload_num_core, slack_time, all ? &barrier_sub[group_id] : &barrier[total_sub_group_id]);
            pthread_barrier_wait(&barrier_finish[total_sub_group_id]);
            if(gemmini_done[other_total_sub_group_id]){
              gemmini_terminate[other_total_sub_group_id] = false;
              inner_done = true;
            }
            if(!inner_done && gemmini_terminate_receive[total_sub_group_id]){
//	      uint64_t planaria_start = read_cycles();
              if(gemmini_terminate[total_sub_group_id]){ // other made it terminate
                pthread_barrier_wait(&barrier_sub_mid2[group_id]);
                gemmini_terminate_receive[total_sub_group_id] = false;
                gemmini_terminate[total_sub_group_id] = false;
		total_queue_planaria_receive[queue_id] ++;
                int other_queue_id = curr_queue_id[other_total_sub_group_id];
                int other_workload_id = total_queue_type[other_queue_id];
                workload_planaria_function(other_queue_id, other_workload_id, workload_num_core + cid, group_id, other_total_sub_group_id, SUB_CORE, 1000, &barrier_sub[group_id]);
#if debug_print == 1
                printf("rid: %d, workload id: %d, queue id: %d terminated - status: %d\n", real_cid, workload_id, queue_id, total_queue_status[queue_id]);
#endif
                pthread_barrier_wait(&barrier_sub_mid3[group_id]);
  //	total_queue_planaria_status[group_id][queue_id] = 1;
              }
              else if(gemmini_terminate[other_total_sub_group_id]){
                pthread_barrier_wait(&barrier_finish[total_sub_group_id]);
                inner_done = gemmini_done[other_total_sub_group_id];
                if(!inner_done){
                   pthread_barrier_wait(&barrier_sub_mid2[group_id]);
               //    total_runtime += workload_planaria_function(queue_id, workload_id, cid, group_id, total_sub_group_id, 4, slack_time, &barrier[nn_args->barrier_index]); // hack: has to be sub barrier
#if debug_print == 1
                   printf("rid: %d, workload id: %d, queue id: %d finished - status: %d\n", real_cid, workload_id, queue_id, total_queue_status[queue_id]);
#endif
	           gemmini_terminate_receive[total_sub_group_id] = false;
                   gemmini_terminate[total_sub_group_id] = false;
                   gemmini_terminate[other_total_sub_group_id] = false; 
	  	   total_queue_planaria_send[queue_id] ++; 
                   total_runtime += workload_planaria_function(queue_id, workload_id, cid, group_id, other_total_sub_group_id, SUB_CORE, 1000, &barrier_sub[group_id]);
                   pthread_barrier_wait(&barrier_sub_mid3[group_id]);
                }
               }
//	     uint64_t planaria_end = read_cycles();
//	     if(cid == 0) total_queue_planaria_time[queue_id] += planaria_end - planaria_start; 
            }
            else if(inner_done && gemmini_terminate_receive[total_sub_group_id]){
              total_runtime += workload_planaria_function(queue_id, workload_id, cid, group_id, total_sub_group_id, 2, 100000000, &barrier[nn_args->barrier_index]); // hack: has to be sub barrier
#if debug_print == 1
              printf("rid: %d, workload id: %d, queue id: %d finished - status: %d\n", real_cid, workload_id, queue_id, total_queue_status[queue_id]);
#endif 
              gemmini_terminate_receive[total_sub_group_id] = false;
              gemmini_terminate[total_sub_group_id] = false;
              pthread_barrier_wait(&barrier[total_sub_group_id]);
            }
            pthread_barrier_wait(&barrier_mid[total_sub_group_id]);
          }
            
          if(workload_num == 0){
            pthread_barrier_wait(&barrier_finish3[total_sub_group_id]);
            if(gemmini_terminate[total_sub_group_id]){
              pthread_barrier_wait(&barrier_sub_mid2[group_id]);
              if(cid == 0) gemmini_done[total_sub_group_id] = true; 
	     //gemmini_fence(); 
	//printf("rid %d residue \n", real_cid);
              gemmini_terminate_receive[total_sub_group_id] = false;
              gemmini_terminate[total_sub_group_id] = false;
	      total_queue_planaria_receive[queue_id] ++;
              int other_queue_id = curr_queue_id[other_total_sub_group_id];
              int other_workload_id = total_queue_type[other_queue_id];
              workload_planaria_function(other_queue_id, other_workload_id, workload_num_core + cid, group_id, other_total_sub_group_id, SUB_CORE, 1000, &barrier_sub[group_id]);
              pthread_barrier_wait(&barrier_sub_mid3[group_id]);
            }
          }
          total_queue_runtime_total[group_cid][queue_id] = total_runtime;
          uint64_t inner_end = read_cycles();
          uint64_t this_cycles = temp_cycles + inner_end - inner_start;
          total_queue_finish[group_cid][queue_id] = (this_cycles > total_queue_dispatch[queue_id]) ? (this_cycles- total_queue_dispatch[queue_id]) : 1000;
          //total_queue_finish[group_cid][queue_id] = ((temp_cycles + inner_end - start) - total_queue_dispatch[queue_id]);

          total_queue_runtime_thread[group_cid][queue_id] = inner_end - inner_start;
          temp_cycles += (inner_end - inner_start);
          pthread_barrier_wait(&barrier_finish2[total_sub_group_id]);


         }
         else total_queue_status[queue_id] = -1; // release the queue 
       }
       else
          break;
     }
   }
   pthread_barrier_wait(&barrier_finish3[total_sub_group_id]);
   if(gemmini_terminate[total_sub_group_id]){
     pthread_barrier_wait(&barrier_sub_mid2[group_id]);
     if(cid == 0) gemmini_done[total_sub_group_id] = true; 
     //gemmini_fence(); 
//printf("rid %d residue \n", real_cid);
     gemmini_terminate_receive[total_sub_group_id] = false;
     gemmini_terminate[total_sub_group_id] = false;
     int other_queue_id = curr_queue_id[other_total_sub_group_id];
     int other_workload_id = total_queue_type[other_queue_id];
     workload_planaria_function(other_queue_id, other_workload_id, workload_num_core + cid, group_id, other_total_sub_group_id, SUB_CORE, 1000, &barrier_sub[group_id]);
     pthread_barrier_wait(&barrier_sub_mid3[group_id]);
   } 
   if(cid == 0) gemmini_done[total_sub_group_id] = true;
   
   pthread_barrier_wait(&barrier_global_end);
   end = read_cycles();
   if(cid == 0) gemmini_done[total_sub_group_id] = false; 
   //if(!done[group_id]){
     if(queue_group != 0)gemmini_runtime[real_cid] += (end - start);
     else gemmini_runtime[real_cid] += 1000000;
     printf("done total sub group id %d rid %d\n", total_sub_group_id, real_cid);
      pthread_barrier_wait(&barrier_sub[group_id]);
     //if(queue_group != 1)
     //   pthread_barrier_wait(&barrier[NUM_CORE]);
     //pthread_barrier_wait(&barrier[NUM_CORE]);
     //printf("idle cycle: %llu\n", gemmini_runtime[real_cid] - temp_cycles);
   //} 
  }
//  printf("finished group id: %d, rid: %d\n", group_id, real_cid);
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
    pthread_barrier_init(&barrier_global_mid, NULL, NUM_CORE);
    pthread_barrier_init(&barrier_global, NULL, NUM_CORE);
    pthread_barrier_init(&barrier_global_start, NULL, NUM_CORE);
    pthread_barrier_init(&barrier_global_end, NULL, NUM_CORE);
 
    printf("starting workload creation \n");
    workload_mode_2(total_workloads, BATCH1, BATCH4, BATCH8, SEED, TARGET_SCALE, CAP_SCALE); 
    printf("workload creation finished \n");

   // int queue_group = 1;
/*
    while((queue_group != 0) || (total_queue_status[total_workloads-1] == -1)){
      global_time = gemmini_runtime[0];
      for(int i = 0; i < NUM_CORE; i++)
        if(global_time < gemmini_runtime[i]) 
          global_time = gemmini_runtime[i];
      queue_group = workload_priority_mp(total_workloads, NUM_ITER, global_time); // or instead use max cycle
      //workload_grouping(queue_group, NUM_GROUP);
      printf("finished workload queue assignment, number of queue group: %d, gemmini runtime: %d\n", queue_group, global_time);

      for(int k = 0; k < NUM_GROUP; k++)
        for(int x = 0; x < queue_group; x++){
           for(int y = 0; y < SUB_GROUP; y++){ 
              printf("group %d queue %d, sub-group %d: ", k, x, y);
              for(int j = 0; j < QUEUE_DEPTH; j++)
                 printf("%d, ", gemmini_workload_assigned[k][y][x][j]);
              printf("\n");
           }
        }
*/
      for(int j = 0; j < NUM_SUB_GROUP; j++)
        gemmini_done[j] = false;

       for(int i = 0; i < NUM_GROUP; i++){
          for(int j = 0; j < SUB_CORE; j++){
            int index = i * SUB_CORE + j;
            nn_args[index].barrier_index = (int)(index / WORKLOAD_CORE);
            nn_args[index].workload_num_core = WORKLOAD_CORE;
            nn_args[index].cid = j % WORKLOAD_CORE;
    //        nn_args[index].queue_group = queue_group;
            nn_args[index].group_id = i;
          }
        }
	for(int i = 0; i < NUM_CORE; i++)
	  pthread_create(&thread[i], &attr[i], thread_NN, &nn_args[i]);	
        for(int i = 0; i < NUM_CORE; i++)
          pthread_join(thread[i], NULL);
      
 

// check total_queue_finish, total_queue_runtime_thread, total_queue_runtime_total of each workload (total_queue_type)
// also check gemmini_runtime 

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
    printf("queue id %d planaria send: %llu\n", i, total_queue_planaria_send[i]);
    printf("queue id %d planaria receive: %llu\n", i, total_queue_planaria_receive[i]);
    printf("queue id %d planaria time: %llu\n", i, total_queue_planaria_time[i]);

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

  exit(0);

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
  pthread_barrier_destroy(&barrier_global_start); 
  pthread_barrier_destroy(&barrier_global_end); 
  pthread_barrier_destroy(&barrier_global_mid); 
  pthread_barrier_destroy(&barrier_global); 
  printf("==================================\n");
  exit(0);
}

