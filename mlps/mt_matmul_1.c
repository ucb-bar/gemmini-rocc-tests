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
#define NUM_ARRAY2 2

#define CHECK_RESULT 1
#define OP 3

#define MAT_DIM_I 512
#define MAT_DIM_J 512
#define MAT_DIM_K 512

pthread_barrier_t barrier;
pthread_barrier_t barrier1;
pthread_barrier_t barrier2;

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



struct thread_args{
    uint64_t cycles, total_cycles;
    int num_array;
};

void *thread_matmul1(void *arg){
    struct thread_args * matmul_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//matmul_args->i;
    uint64_t total_start = read_cycles();
    for(int i = 0; i < NUM_ARRAY1; i++)
      while(!rerocc_acquire(i, 0xf)){}

    for (int i = 0; i < NUM_ARRAY1; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }
    
    pthread_barrier_wait(&barrier);

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
            WS, NUM_ARRAY1, 0);
    uint64_t end = read_cycles();
    matmul_args->cycles = end - start;
 
    for(int i = 0; i < NUM_ARRAY1; i++)
      rerocc_release(i);   
    uint64_t total_end = read_cycles();
    matmul_args->total_cycles = total_end - total_start;
}

void *thread_matmul2(void *arg){
    struct thread_args * matmul_args = (struct thread_args *) arg;
    int cid = sched_getcpu();//matmul_args->i;
    uint64_t total_start =  read_cycles();
    for(int i = 0; i < NUM_ARRAY2; i++)
      while(!rerocc_acquire(i, 0xf)){}

    for (int i = 0; i < NUM_ARRAY2; i++) {
      rerocc_assign(OP, i);
      gemmini_flush(0);
    }
    
    pthread_barrier_wait(&barrier);

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
            WS, NUM_ARRAY2, 0);
    uint64_t end = read_cycles();
    matmul_args->cycles = end - start;
 
    for(int i = 0; i < NUM_ARRAY2; i++)
      rerocc_release(i);   
    uint64_t total_end = read_cycles();
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
      printf("Cycles taken: %llu, %llu\n", matmul_args[i].cycles, matmul_args[i].total_cycles);
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
