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
#define NO_BIAS false
#define NUM_ARRAY1 2
#define NUM_ARRAY2 4
#define TOTAL_ARRAY 5 // number of total array available

#define THREAD_PRINT 1
#define CHECK_RESULT 1
#define OP 3

#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512

pthread_barrier_t barrier;
pthread_barrier_t barrier1;
pthread_mutex_t array_mutex;
pthread_mutex_t num_mutex;

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

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], acc_t D[MAT_DIM_I][MAT_DIM_J], elem_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = (elem_t)(D[r][c]);
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}


 
static elem_t A1[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B1[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static acc_t D1[MAT_DIM_I][MAT_DIM_J] row_align_acc(1) = {0};
static elem_t Out1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t gold1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static elem_t A2[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN) = {0};
static elem_t B2[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static acc_t D2[MAT_DIM_I][MAT_DIM_J] row_align_acc(2) = {0};
static elem_t Out2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};
static elem_t gold2[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN) = {0};

static int num_available_array = 0;

struct thread_args{
    uint64_t cycles, total_cycles;
    int num_array;
    int num_mutex_count, num_array_used; // for check purpose
};

void *thread_matmul1(void *arg){
    struct thread_args * matmul_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//matmul_args->i;
    int request_array = 0;//matmul_args->num_array;
    pthread_mutex_lock(&array_mutex);
#if THREAD_PRINT == 1
    printf("cid %d got lock\n", cid);
#endif
    bool num_mutex_pass = false;
    int num_mutex_count = 0;
    while(!num_mutex_pass){
      pthread_mutex_lock(&num_mutex);
      if(num_available_array > 0){
        request_array = matmul_args->num_array;
        if(request_array > num_available_array && num_available_array > 0){
          request_array = num_available_array;
          //num_available_array = 0;//-= request_array;
        }
        num_available_array -= request_array;
        num_mutex_pass = true;
      }
      else{
        num_mutex_pass = false;
      }
      pthread_mutex_unlock(&num_mutex);
      if(!num_mutex_pass){
        uint64_t sleep_start = read_cycles();
        uint64_t sleep_end = read_cycles();
        while(sleep_end - sleep_start > 10000){
          sleep_end = read_cycles();
        }
        num_mutex_count ++;
      }
    }
#if THREA_PRINT == 1
    printf("number of num_array_mutex turn: %d, requested array: %d\n", num_mutex_count, request_array);
#endif
    uint64_t total_start = read_cycles();
    for(int i = 0; i < request_array; i++)
      while(!rerocc_acquire(i, 0xf)){}
    pthread_mutex_unlock(&array_mutex);
#if THREAD_PRINT == 1
    printf("cid %d release lock\n", cid);
#endif
    for (int i = 0; i < request_array; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }
    
	uint64_t start = read_cycles();
    tiled_opcode_matmul_nn_auto_multi(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, // strides
            false, false, false, false, // direct dram?
            (elem_t*)A1, (elem_t*)B1, NO_BIAS ? NULL : &D1[0][0], (elem_t*)Out1,
            //MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            //MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            //false, false,
            //false, !FULL_BIAS_WIDTH,
            //0,
            WS, request_array, 0);
    uint64_t end = read_cycles();
    matmul_args->cycles = end - start;
 
#if THREAD_PRINT == 1
    printf("cid %d finish operation\n", cid);
#endif
    for(int i = 0; i < request_array; i++)
      rerocc_release(i);  
    pthread_mutex_lock(&num_mutex);
    num_available_array += request_array;
    pthread_mutex_unlock(&num_mutex);
  
    uint64_t total_end = read_cycles();
#if THREAD_PRINT == 1
    printf("cid %d release rerocc \n", cid);
#endif
    matmul_args->num_mutex_count = num_mutex_count;
    matmul_args->num_array_used = request_array;
    matmul_args->total_cycles = total_end - total_start;
}

void *thread_matmul2(void *arg){
    struct thread_args * matmul_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//matmul_args->i;
    int request_array = 0;//matmul_args->num_array;
    pthread_mutex_lock(&array_mutex);
#if THREAD_PRINT == 1
    printf("cid %d got lock\n", cid);
#endif

    bool num_mutex_pass = false;
    int num_mutex_count = 0;
    while(!num_mutex_pass){
      pthread_mutex_lock(&num_mutex);
      if(num_available_array > 0){
        if(request_array > num_available_array && num_available_array > 0){
          request_array = num_available_array;
          //num_available_array = 0;//-= request_array;
        }
        num_available_array -= request_array;
        num_mutex_pass = true;
      }
      else{
        num_mutex_pass = false;
      }
      pthread_mutex_unlock(&num_mutex);
      if(!num_mutex_pass){
        uint64_t sleep_start = read_cycles();
        uint64_t sleep_end = read_cycles();
        while(sleep_end - sleep_start > 10000){
          sleep_end = read_cycles();
        }
        num_mutex_count ++;
      }
    }
#if THREA_PRINT == 1
    printf("number of num_array_mutex turn: %d, requested array: %d\n", num_mutex_count, request_array);
#endif
    uint64_t total_start = read_cycles();
    for(int i = 0; i < request_array; i++)
      while(!rerocc_acquire(i, 0xf)){}
    pthread_mutex_unlock(&array_mutex);
#if THREAD_PRINT == 1
    printf("cid %d release lock\n", cid);
#endif
    for (int i = 0; i < request_array; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }
    
	uint64_t start = read_cycles();
    tiled_opcode_matmul_nn_auto_multi(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, // strides
            false, false, false, false, // direct dram?
            (elem_t*)A2, (elem_t*)B2, NO_BIAS ? NULL : &D2[0][0], (elem_t*)Out2,
            //MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            //MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            //false, false,
            //false, !FULL_BIAS_WIDTH,
            //0,
            WS, request_array, 0);
    uint64_t end = read_cycles();
    matmul_args->cycles = end - start;
 
#if THREAD_PRINT == 1
    printf("cid %d finish operation\n", cid);
#endif
    for(int i = 0; i < request_array; i++)
      rerocc_release(i);
    pthread_mutex_lock(&num_mutex);
    num_available_array += request_array;
    pthread_mutex_unlock(&num_mutex);   
    uint64_t total_end = read_cycles();
#if THREAD_PRINT == 1
    printf("cid %d release rerocc \n", cid);
#endif
    matmul_args->num_mutex_count = num_mutex_count;
    matmul_args->num_array_used = request_array;
    matmul_args->total_cycles = total_end - total_start;
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
     struct thread_args matmul_args[num_proc];
     for(int i = 0; i < num_proc; i++){
             CPU_ZERO(&cpuset[i]);
             CPU_SET(i, &cpuset[i]);
             pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
             pthread_create(&thread[i], &attr[i], print_message, NULL);
     }

     for(int i = 0; i < num_proc; i++){
            pthread_join(thread[i], NULL);
     }

     num_available_array = TOTAL_ARRAY;

#if CHECK_RESULT == 1
    printf("Init A and B\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        A1[i][j] = (rand() % 3)-1;
        B1[i][j] = (rand() % 3)-1;
        A2[i][j] = (rand() % 3)-1;
        B2[i][j] = (rand() % 3)-1;
        D1[i][j] = NO_BIAS  ? 0 : (rand() % 2);
        D2[i][j] = NO_BIAS  ? 0 : (rand() % 2);
      }
    }

    full_matmul(A1, B1, D1, gold1);
    full_matmul(A2, B2, D2, gold2);
#endif
    if (pthread_mutex_init(&array_mutex, NULL) != 0){
      printf("\n mutex init failed\n");
      return 1;
    }  
    if (pthread_mutex_init(&num_mutex, NULL) != 0){
      printf("\n mutex init failed\n");
      return 1;
    }
    pthread_barrier_init(&barrier, NULL, num_proc);
    for(int i = 0; i < num_proc; i++){
      if(i == 0){
        matmul_args[i].num_array = NUM_ARRAY1;
        pthread_create(&thread[i], &attr[i], thread_matmul1, &matmul_args[i]);
      }
      else{
        matmul_args[i].num_array = NUM_ARRAY2;
        pthread_create(&thread[i], &attr[i], thread_matmul2, &matmul_args[i]);
      }
    }

    for(int i = 0; i < num_proc; i++)
      pthread_join(thread[i], NULL);


    for(int i = 0; i < num_proc; i++){
      printf("cid %d Cycles taken: %llu, %llu\n", i, matmul_args[i].cycles, matmul_args[i].total_cycles);
      printf("cid %d number of num_array_mutex turn: %d, requested array: %d\n", i, matmul_args[i].num_mutex_count, matmul_args[i].num_array_used);
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
