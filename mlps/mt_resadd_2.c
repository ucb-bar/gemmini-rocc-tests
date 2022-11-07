// See LICENSE for license details.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

// in common
#define num_proc 2
#define A_SCALE 1
#define B_SCALE MVIN_SCALE_IDENTITY
#define C_SCALE ACC_SCALE_IDENTITY
#define USE_RELU false
#define NUM_ARRAY1 4
#define NUM_ARRAY2 2

#define CHECK_RESULT 1
#define OP 3

#define MAT_DIM_I 256
#define MAT_DIM_J 256

pthread_barrier_t barrier;
pthread_barrier_t barrier1;
pthread_mutex_t array_mutex;

static size_t args[7];
void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}


void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j]){
        printf("i: %d, j: %d, x: %d, y: %d\n", i, j, x[i][j], y[i][j]); 
        //return 0;
      }
  return 1;
}


 
static elem_t A1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
//static acc_t bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1) = {0};
static elem_t Out1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t gold1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static elem_t A2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
//static acc_t bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(2) = {0};
static elem_t Out2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t gold2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};



struct thread_args{
    uint64_t cycles, total_cycles;
    int num_array;
};

void *thread_resadd1(void *arg){
    struct thread_args * resadd_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//resadd_args->i;

    pthread_mutex_lock(&array_mutex);
    printf("cid %d got lock\n", cid);

    uint64_t total_start = read_cycles();
    for(int i = 0; i < NUM_ARRAY1; i++)
      while(!rerocc_acquire(i, 0xf)){}
    pthread_mutex_unlock(&array_mutex);
    printf("cid %d release lock\n", cid);

    for (int i = 0; i < NUM_ARRAY1; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }

	uint64_t start = read_cycles();
    tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
            (elem_t*)A1, (elem_t*)B1,
            (elem_t*)Out1, USE_RELU, WS, NUM_ARRAY1, 0);
    uint64_t end = read_cycles();
    resadd_args->cycles = end - start;
 
    printf("cid %d finish operation\n", cid);
    for(int i = 0; i < NUM_ARRAY1; i++)
      rerocc_release(i);  
    uint64_t total_end = read_cycles();
    printf("cid %d release rerocc \n", cid);
    resadd_args->total_cycles = total_end - total_start;
}

void *thread_resadd2(void *arg){
    struct thread_args * resadd_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//resadd_args->i;
    
    pthread_mutex_lock(&array_mutex);
    printf("cid %d got lock\n", cid);

    uint64_t total_start =  read_cycles();
    for(int i = 0; i < NUM_ARRAY2; i++)
      while(!rerocc_acquire(i, 0xf)){}
    pthread_mutex_unlock(&array_mutex);
    printf("cid %d release lock\n", cid);

    for (int i = 0; i < NUM_ARRAY2; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }
    
	uint64_t start = read_cycles();
    tiled_opcode_resadd_auto_multi(MAT_DIM_I, MAT_DIM_J, A_SCALE, B_SCALE, C_SCALE, MAT_DIM_J, false, false, false,
            (elem_t*)A2, (elem_t*)B2,
            (elem_t*)Out2, USE_RELU, WS, NUM_ARRAY2, 0);
    uint64_t end = read_cycles();
    resadd_args->cycles = end - start;
    printf("cid %d finish operation\n", cid);
 
    for(int i = 0; i < NUM_ARRAY2; i++)
      rerocc_release(i);   
    uint64_t total_end = read_cycles();
    printf("cid %d release rerocc \n", cid);
    resadd_args->total_cycles = total_end - total_start;
}


void *print_message(void *ptr){
    int cpu_id = sched_getcpu();
   // char *msg;
   // msg = (char *) ptr;
    printf("print msg - cpu_id: %d \n", cpu_id);
   // printf("%s \n", msg);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif
     int cpu_id;
     cpu_id = sched_getcpu();
     printf("main thread cpuid: %d \n", cpu_id);
     cpu_set_t cpuset[num_proc];
     pthread_t thread[num_proc];
     pthread_attr_t attr[num_proc];
     for(int i = 0; i < num_proc; i++)
            pthread_attr_init(&attr[i]);
     struct thread_args resadd_args[num_proc];
     for(int i = 0; i < num_proc; i++){
             CPU_ZERO(&cpuset[i]);
             CPU_SET(i, &cpuset[i]);
             pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
             pthread_create(&thread[i], &attr[i], print_message, NULL);
     }

     for(int i = 0; i < num_proc; i++){
            pthread_join(thread[i], NULL);
     }

#if CHECK_RESULT == 1
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A1[i][j] = (rand() % 8);
        B1[i][j] = (rand() % 3);
        gold1[i][j] = A1[i][j]+B1[i][j];
        A2[i][j] = (rand() % 8);
        B2[i][j] = (rand() % 3);
        gold2[i][j] = A2[i][j]+B2[i][j];
      }
    }

#endif  
    if (pthread_mutex_init(&array_mutex, NULL) != 0){
      printf("\n mutex init failed\n");
      return 1;
    }
    pthread_barrier_init(&barrier, NULL, num_proc);
    for(int i = 0; i < num_proc; i++){
      if(i == 0){
        resadd_args[i].num_array = NUM_ARRAY1;
        pthread_create(&thread[i], &attr[i], thread_resadd1, &resadd_args[i]);
      }
      else{
        resadd_args[i].num_array = NUM_ARRAY2;
        pthread_create(&thread[i], &attr[i], thread_resadd2, &resadd_args[i]);
      }
    }

    for(int i = 0; i < num_proc; i++)
      pthread_join(thread[i], NULL);


    for(int i = 0; i < num_proc; i++){
      printf("Cycles taken: %llu, %llu\n", resadd_args[i].cycles, resadd_args[i].total_cycles);
    }

	pthread_barrier_destroy(&barrier);
	//pthread_barrier_destroy(&barrier2);

#if CHECK_RESULT == 1
    for(int j = 0; j < num_proc; j++){
      if(j == 0) {
          printf("result for 1 \n");
          full_is_equal(Out1, gold1);
      }
      else if(j == 1) {
          printf("result for 2 \n");
          full_is_equal(Out2, gold2);
      }
    }
#endif
}
