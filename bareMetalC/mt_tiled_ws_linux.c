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
#define CHECK_RESULT 0

#define NO_BIAS 1
#define REPEATING_BIAS 0
#define FULL_BIAS_WIDTH 1

#define A_TRANSPOSE 0
#define B_TRANSPOSE 0



#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#error variable-bitwidth bias not currently supported
#endif

#ifndef BAREMETAL
#define MAT_DIM 1024
#define MAT_DIM_I MAT_DIM
#define MAT_DIM_K MAT_DIM
#define MAT_DIM_J MAT_DIM
#else
#define MAT_DIM 32
#define MAT_DIM_I MAT_DIM
#define MAT_DIM_K MAT_DIM
#define MAT_DIM_J MAT_DIM
#endif
#define num_proc 4

#if A_TRANSPOSE==0
#define A_STRIDE MAT_DIM_K
#else
#define A_STRIDE MAT_DIM_I
#endif

#if B_TRANSPOSE==0
#define B_STRIDE MAT_DIM_J
#else
#define B_STRIDE MAT_DIM_K
#endif


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
      if (x[i][j] != y[i][j])
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
static elem_t in_A[MAT_DIM_I][MAT_DIM_K] row_align(1) = {0};
static elem_t in_B[MAT_DIM_K][MAT_DIM_J] row_align(1) = {0};
//static ACC_T bias[MAT_DIM_I][MAT_DIM_J] row_align_acc(1) = {0};
static elem_t Out[num_proc][MAT_DIM_I][MAT_DIM_J] row_align(1) = {0};

struct thread_args{
	int i;
};

void *thread_matmul(void *arg){
	struct thread_args * matmul_args = (struct thread_args *) arg;
	gemmini_flush(0);
	int cid = matmul_args->i;
	int b_unit = MAX_BLOCK_LEN;
	  elem_t* A = (elem_t*) in_A + MAT_DIM_K*DIM*(cid/2);
	  elem_t* B = (elem_t*) in_B + b_unit*DIM*(cid%2);
	  elem_t* C = (elem_t*) Out + b_unit*DIM*(cid%2) + MAT_DIM_J*DIM*(cid/2);
//	  acc_t * D = (acc_t*) bias + b_unit*DIM*(cid%2) + MAT_DIM_J*DIM*(cid/2);
	 
   uint64_t start = read_cycles(); 
	  tiled_matmul_auto_distance(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K, DIM*num_proc/2, DIM*num_proc/2*b_unit, 2, 2,
				A, B, NULL, C, //NO_BIAS ? NULL : D, C,
			   A_STRIDE, B_STRIDE, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            A_TRANSPOSE, B_TRANSPOSE,
            WS);

    uint64_t end = read_cycles();
    printf("CPU %d Cycles taken: %u\n", sched_getcpu(), end-start);
    const uint64_t total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K / num_proc;
    const uint64_t ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("CPU %d Utilization: %d%%\n", sched_getcpu(), utilization);
	
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
#if CHECK_RESULT == 1
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

    // printf("Init A\n");
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

    printf("Starting slow CPU matmul\n");
    unsigned long cpu_start = read_cycles();
    full_matmul(in_A, in_B, bias, gold_full);
    unsigned long cpu_end = read_cycles();
    printf("Cycles taken: %u\n", cpu_end-cpu_start);
    full_matshift(gold_full, gold, 0);
#endif

#ifndef BAREMETAL
	 int cpu_id;
	 cpu_id = sched_getcpu();
	 cpu_set_t cpuset[num_proc];
	 pthread_t thread[num_proc];
	 pthread_attr_t attr;
	 pthread_attr_init(&attr);
	 struct thread_args matmul_args[num_proc];


	 for(int i = 0; i < num_proc; i++){
		matmul_args[i].i = i;
		 CPU_ZERO(&cpuset[i]);
		 CPU_SET(i, &cpuset[i]);
		 pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[i]);
		 pthread_create(&thread[i], &attr, print_message, NULL);
	 }

/*	 for(int i = 0; i < num_proc; i++){
		pthread_join(thread[i], NULL);
	 }
*/
	 pthread_join(thread[0], NULL);
	 pthread_join(thread[1], NULL); 
	 pthread_join(thread[2], NULL);
	 pthread_join(thread[3], NULL);
	//start gemmini matmul cycle count
	//unsigned long start = read_cycles();
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[0]);
	pthread_create(&thread[0], &attr, thread_matmul, &matmul_args[0]);
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[1]);
	pthread_create(&thread[1], &attr, thread_matmul, &matmul_args[1]);
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[2]);
	pthread_create(&thread[2], &attr, thread_matmul, &matmul_args[2]);
	pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[3]);
	pthread_create(&thread[3], &attr, thread_matmul, &matmul_args[3]);
/*
	for(int i = 0; i < num_proc; i++){
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset[i]);
		pthread_create(&thread[i], &attr, thread_matmul, &matmul_args[i]);
	}
*/
	for(int i = 0; i < num_proc; i++)
		pthread_join(thread[i], NULL);

/*
    unsigned long end = read_cycles();
    printf("Cycles taken: %u\n", end-start);
    const int total_macs = MAT_DIM_I * MAT_DIM_J * MAT_DIM_K / num_proc;
    const int ideal_cycles = total_macs / (DIM * DIM);
    const int utilization = 100 * ideal_cycles / (end-start);
    printf("Utilization: %d%%\n", utilization);
*/	
#endif

#if CHECK_RESULT == 1
    if (!full_is_equal(Out, gold)) {
      printf("C:\n");
      full_printMatrix(Out);
      printf("Gold:\n");
      full_printMatrix(gold);
      printf("\n");

      exit(1);
    }
#endif

  exit(0);
}

