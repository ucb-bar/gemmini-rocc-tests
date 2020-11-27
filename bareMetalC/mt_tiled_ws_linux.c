// See LICENSE for license details.
#define _GNU_SOURCE
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
#include "util.h"
#include <pthread.h>

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#ifndef BAREMETAL
#define MAT_DIM 128
#define MAT_DIM_I MAT_DIM
#define MAT_DIM_K MAT_DIM
#define MAT_DIM_J MAT_DIM
#else
#define MAT_DIM 32
#define MAT_DIM_I MAT_DIM
#define MAT_DIM_K MAT_DIM
#define MAT_DIM_J MAT_DIM
#endif
#define num_proc 2

void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t* m) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", *(m+i*MAT_DIM_J+j));
    printf("\n");
  }
}

int full_is_equal(elem_t* x, elem_t* y) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (*(x+i*MAT_DIM_J+j) != *(y+i*MAT_DIM_J+j))
        return 0;
  return 1;
}

void full_matshift(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], int shift) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Bitshift and round element
      full_t shifted = ROUNDING_RIGHT_SHIFT(full[r][c], shift);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = shifted > elem_t_max ? elem_t_max : (shifted < elem_t_min ? elem_t_min : shifted);
      out[r][c] = elem;
#else
      out[r][c] = shifted; // TODO should we also saturate when using floats?
#endif
    }
} 

//start from here
static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
static ACC_T bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
static elem_t Out[num_proc][MAT_DIM_I][MAT_DIM_J] row_align(1);
static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
static elem_t gold[MAT_DIM_I][MAT_DIM_J];


struct thread_args{
	int start_I;//, start_J, start_K;
	int dim_I, dim_J;//, dim_K;
	bool first;
};

void *init(void *arg){
    int cpu_id;
    cpu_id = sched_getcpu();
    printf("init-cpu_id: %d \n");
	
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        in_A[i][j] = rand() % 2;
      }
    }

    // printf("Init B\n");
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        in_B[i][j] = rand() % 2;
      }
    }

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        bias[i][j] = NO_BIAS ? 0 : rand() % 2;
      }
    }
}

void *cpu_matmul(void *arg){
    int cpu_id = sched_getcpu();
    printf("cpu_matmul - cpu_id: %d \n", cpu_id);
    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();
    full_matmul(in_A, in_B, bias, gold_full);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    full_matshift(gold_full, gold, 0);
}



void *thread_matmul(void *arg){
	struct thread_args * matmul_args = (struct thread_args *) arg;
	if(matmul_args->first) gemmini_flush(0);

	int cpu_id;
	cpu_id = sched_getcpu();
	int start_J = cpu_id == 0 ? 0 : matmul_args->dim_J;
	int dim_J = cpu_id == 0 ? matmul_args->dim_J : MAT_DIM - start_J;

	elem_t* in_A_start = (elem_t*) in_A + MAT_DIM_K*(matmul_args->start_I);
	elem_t* in_B_start = (elem_t*) in_B + start_J;
	elem_t* Out_start = (elem_t*) Out + MAT_DIM_J*(matmul_args->start_I)+start_J;
	ACC_T* bias_start = (ACC_T*) bias + MAT_DIM_J*(matmul_args->start_I)+start_J;

	printf("cpu_id: %d, start I, J: %d, %d \n", cpu_id, matmul_args->start_I, start_J);
	printf("dim I, J: %d, %d \n", matmul_args->dim_I, dim_J);

	tiled_matmul_auto(matmul_args->dim_I, dim_J, MAT_DIM_K,
			in_A_start, in_B_start, NO_BIAS? NULL : bias_start, Out_start,
			MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
			MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
			NO_ACTIVATION, 0, 0, false,
			WS);

}

void *print_message(void *ptr){
	int cpu_id = sched_getcpu();
	char *msg;
	msg = (char *) ptr;
	printf("print msg - cpu_id: %d \n", cpu_id);
	printf("%s \n", msg);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#if CHECK_RESULT == 1
    pthread_t thread0, thread1;
    char *msg0 = "thread0";
    char *msg1  = "thread1";
    printf("initializing \n");
    pthread_create(&thread0, NULL, init, NULL);	
    pthread_create(&thread1, NULL, print_message, (void *) msg1);
    pthread_detach(thread0);
    pthread_detach(thread1);

    printf("multiplying \n");
    pthread_create(&thread0, NULL, cpu_matmul, NULL);
    pthread_create(&thread1, NULL, print_message, (void*) msg1);
    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);
    printf("finished initial steps \n");
	
#endif

#ifndef BAREMETAL

	 cpu_set_t cpuset;
	 pthread_t thread[num_proc];
	 //int thread1_create, thread2_create;

	 pthread_attr_t attr;
	 pthread_attr_init(&attr);
	 struct thread_args matmul_args[num_proc];
	 int current_I;//, current_J, current_K;
	 int range; //current starting point, range of matmul
	 current_I = 0;
	 //current_J = 0;
	 //divide matrix into 2x2 block
	 range = MAT_DIM / num_proc; //for now, assume square

	 //start gemmini matmul cycle count
	 printf("Starting gemmini matmul\n");
         unsigned long start = read_cycles();

	 for(int j = 0; j < 2; j++){ //there are 2 row blocks
		 for(int i = 0; i < num_proc; i++){
			 matmul_args[i].first = (j==0); //for gemmini_flush
			 matmul_args[i].start_I = current_I;
			 //matmul_args[i].start_J = current_J;
			 //matmul_args[i].start_K = 0;
			 matmul_args[i].dim_I = (range + current_I) > MAT_DIM ? MAT_DIM - current_I : range;
			 matmul_args[i].dim_J = range;//(range + current_J) > MAT_DIM ? MAT_DIM - current_J : range;
			 //matmul_args[i].dim_K = MAT_DIM_K;
			 current_I += (i == num_proc-1) ? matmul_args[i].dim_I : 0; //update starting point
			 //current_J = (i == num_proc-1) ? 0 : matmul_args[i].dim_J;
			 //printf("loop i: %d, j: %d, current_I: %d, current_J: %d \n", i, j, current_I, current_J);

			 CPU_ZERO(&cpuset); //empty the cpu set
			 CPU_SET(i, &cpuset); //add each cpu to cpu set
			 pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
			 pthread_create(&thread[i], &attr, thread_matmul, &matmul_args);	
		 }
/*
		 for(int i = 0; i < num_proc;i++){
			 CPU_ZERO(&cpuset); //empty the cpu set
			 CPU_SET(i, &cpuset); //add each cpu to cpu set
			 pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
			 pthread_create(&thread[i], &attr, thread_matmul, &matmul_args);
		 }
*/		
		 for(int i = 0; i < num_proc; i++)
			 pthread_join(thread[i], NULL);
	 }

    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
#endif

#if CHECK_RESULT == 1
    if (!full_is_equal((elem_t*) Out, (elem_t*) gold)) {
      printf("C:\n");
      full_printMatrix((elem_t*) Out);
      printf("Gold:\n");
      full_printMatrix((elem_t*) gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

